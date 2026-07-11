# Deploying to Azure Container Apps

Single Docker image: FastAPI (`backend/main.py`) serves both the `/api/*` routes
and the built React SPA at `/`. These steps build the image in Azure Container
Registry (ACR) and run it on Azure Container Apps (ACA).

## Prerequisites

- `az` CLI logged in (`az login`) with a subscription selected.
- A resource group, e.g. `az group create -n patho-rag -l eastus`.
- Your `GOOGLE_API_KEY` for Gemini.
- **Vector DB present locally**: `output/biomedbert_vector_db/{faiss.index,metadata.pkl}`
  are tracked with git-LFS. Run `git lfs pull` so the image bakes real files,
  not LFS pointer stubs (otherwise the app boots "degraded").

## 1. Create an ACR and build the image

```bash
az acr create -g patho-rag -n pathoragacr --sku Basic
az acr build -r pathoragacr -t pathorag:latest .   # builds Dockerfile in the cloud
```

## 2. Create the Container Apps environment

```bash
az extension add --name containerapp --upgrade
az containerapp env create -g patho-rag -n patho-env -l eastus
```

## 3. Create the app

```bash
az containerapp create \
  -g patho-rag -n pathorag \
  --environment patho-env \
  --image pathoragacr.azurecr.io/pathorag:latest \
  --registry-server pathoragacr.azurecr.io \
  --target-port 8000 --ingress external \
  --min-replicas 1 --max-replicas 3 \
  --cpu 2 --memory 4Gi \
  --secrets google-api-key=<YOUR_GEMINI_KEY> \
  --env-vars GOOGLE_API_KEY=secretref:google-api-key
```

- `--min-replicas 1` avoids scale-to-zero so the models stay warm (cold start
  reloads BiomedBERT + cross-encoder + PaddleOCR, which is slow).
- Bump `--cpu/--memory` if OCR/embedding is heavy; torch + paddle want headroom.

The public URL is the app's FQDN:

```bash
az containerapp show -g patho-rag -n pathorag --query properties.configuration.ingress.fqdn -o tsv
```

## 4. Persist uploads + index across restarts (important)

The container filesystem is **ephemeral** — uploaded PDFs and the FAISS index
updates from new documents are lost on restart/redeploy unless mounted.

1. Create a storage account + file share, and register it with the ACA env:

```bash
az storage account create -g patho-rag -n pathoragstore --sku Standard_LRS
az storage share create --account-name pathoragstore -n ragdata
az containerapp env storage set -g patho-rag -n patho-env \
  --storage-name ragdata --azure-file-account-name pathoragstore \
  --azure-file-account-key <KEY> --azure-file-share-name ragdata \
  --access-mode ReadWrite
```

2. Mount it into the app (via `az containerapp update --yaml` volume/volumeMounts)
   at both `uploaded_reports/` and `output/biomedbert_vector_db/`, and seed the
   share with the initial `output/biomedbert_vector_db/` contents so the base
   corpus is available on first boot.

## 5. Model cache (optional, faster cold starts)

BiomedBERT, the cross-encoder, and PaddleOCR weights download on first use. To
avoid re-downloading on every new replica, either bake a warm cache into the
image (pre-download during `docker build`) or point `HF_HOME` / the paddle cache
dir at the mounted Azure Files share.

## Update / redeploy

```bash
az acr build -r pathoragacr -t pathorag:latest .
az containerapp update -g patho-rag -n pathorag \
  --image pathoragacr.azurecr.io/pathorag:latest
```

## Smoke test

- Open the FQDN → the SPA loads.
- `GET /api/health` → `{"status":"ok","num_documents":N}`.
- Ask a question → answer streams token-by-token with citations.
- Upload a PDF → progress bar → the doc becomes selectable and highlights work.
- Restart a replica → previously uploaded docs still present (confirms the mount).
```
