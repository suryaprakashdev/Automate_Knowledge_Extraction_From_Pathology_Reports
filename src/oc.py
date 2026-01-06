#!/usr/bin/env python3
"""
Direct GDC PDF OCR without downloading - streams PDFs from GDC API
"""

from pathlib import Path
from tqdm import tqdm
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import requests
import io
import time

# Paths
OUTPUT_DIR = Path("paddleocr_results/text")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize OCR once
ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=True,
    device="gpu"
)


def download_pdf_to_memory(file_id):
    """Download PDF from GDC API directly to memory."""
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Read PDF into memory
        pdf_bytes = io.BytesIO()
        for chunk in response.iter_content(chunk_size=1024*1024):
            if chunk:
                pdf_bytes.write(chunk)
        
        pdf_bytes.seek(0)
        return pdf_bytes
        
    except Exception as e:
        raise RuntimeError(f"Failed to download: {e}")


def pdf_to_images(pdf_bytes):
    """Convert PDF bytes to PIL images."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images


def ocr_pdf_from_gdc(file_id):
    """Download PDF from GDC and perform OCR."""
    # Download PDF to memory
    pdf_bytes = download_pdf_to_memory(file_id)
    
    # Convert to images
    images = pdf_to_images(pdf_bytes)
    all_text = []

    for page_idx, img in enumerate(images, 1):
        img_np = np.array(img)

        result = ocr.ocr(img_np, cls=True)
        if result and result[0]:
            page_text = " ".join([line[1][0] for line in result[0]])
            all_text.append(f"\n--- Page {page_idx} ---\n{page_text}")

    return "\n".join(all_text)


def main():
    # Read manifest file
    manifest_file = "/usr/users/3d_dimension_est/selva_sur/RAG/data/file_ids.txt" 
    
    if not Path(manifest_file).exists():
        print(f"❌ ERROR: Manifest file not found: {manifest_file}")
        return
    
    with open(manifest_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    
    file_ids = [line.split('\t')[0].strip() for line in lines if line.strip()]
    
    # Process all files (or limit for testing)
    # file_ids = file_ids[:100]  # Uncomment to test with first 100
    
    print("\n" + "=" * 70)
    print("PADDLEOCR – DIRECT GDC PDF OCR (NO DOWNLOAD)")
    print("=" * 70)
    print(f"Manifest file    : {manifest_file}")
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"Files to process : {len(file_ids):,}")
    print(f"Device           : GPU")
    print("=" * 70 + "\n")

    success, failed = 0, 0
    total_chars = 0
    failed_ids = []

    for file_id in tqdm(file_ids, desc="Processing PDFs", unit="files"):
        try:
            # OCR the PDF directly from GDC
            text = ocr_pdf_from_gdc(file_id)

            if not text.strip():
                raise RuntimeError("No text extracted")

            # Save output
            out_file = OUTPUT_DIR / f"{file_id}.txt"
            out_file.write_text(text, encoding="utf-8")

            total_chars += len(text)
            success += 1

        except Exception as e:
            failed += 1
            failed_ids.append(file_id)
            tqdm.write(f"❌ {file_id}: {e}")
        
        # Small delay to avoid hammering the API
        time.sleep(0.1)

    print("\n" + "=" * 70)
    print("OCR SUMMARY")
    print("=" * 70)
    print(f"Total attempted : {len(file_ids):,}")
    print(f"Successful      : {success:,}")
    print(f"Failed          : {failed:,}")
    print(f"Success rate    : {100 * success / max(len(file_ids), 1):.1f}%")
    print(f"Total text      : {total_chars:,} chars")
    print("=" * 70)
    
    # Save failed IDs
    if failed_ids:
        failed_file = OUTPUT_DIR / "failed_ocr_ids.txt"
        with open(failed_file, 'w') as f:
            f.write("file_id\n")
            for fid in failed_ids:
                f.write(f"{fid}\n")
        print(f"\n⚠️  Failed file IDs saved to: {failed_file}")


if __name__ == "__main__":
    main()