# ── Stage 1: build the React frontend ───────────────────────────────────────
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build            # emits /app/frontend/dist

# ── Stage 2: python runtime (FastAPI serves API + built SPA) ─────────────────
FROM python:3.11-slim AS runtime
WORKDIR /app

# Build deps for faiss / paddle / native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps first (better layer caching)
COPY requirements.txt ./
RUN pip install "torch>=2.3" --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt


# App code + the base vector DB
COPY src/ ./src/
COPY backend/ ./backend/
COPY output/ ./output/

# Built frontend from stage 1
COPY --from=frontend /app/frontend/dist ./frontend/dist

ENV CUDA_VISIBLE_DEVICES="" \
    DB_PATH=output/biomedbert_vector_db \
    UPLOAD_DIR=uploaded_reports \
    PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
