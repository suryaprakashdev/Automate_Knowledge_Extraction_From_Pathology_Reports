"""
Multimodal Document Preprocessor
================================
Complete preprocessing pipeline for PDFs, Images, and Audio files.
PDF Text → PDF to Image → OCR → Image Extraction → Chunking → Testing
"""

import os
import logging
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import librosa
import soundfile as sf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MultimodalPreprocessor:
    """
    Complete document preprocessor for multimodal RAG system.
    Processes PDFs, images, and audio files into LangChain Documents.
    """
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    extract_images: bool = True
    ocr_enabled: bool = True
    process_audio_enabled: bool = True
    enhance_images: bool = True
    output_dir: str = "preprocessed_outputs"
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize output directories for processed files."""
        self.image_dir = os.path.join(self.output_dir, "images")
        self.audio_dir = os.path.join(self.output_dir, "audio")
        self.enhanced_dir = os.path.join(self.output_dir, "enhanced")
        
        for directory in [self.image_dir, self.audio_dir, self.enhanced_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ Initialized preprocessor with output dir: {self.output_dir}")
    
    
    # ============================================================================
    # STEP 1: IMPLEMENT PDF TEXT EXTRACTION
    # ============================================================================
    
    def extract_text_from_pdf(self, file_path: str) -> Tuple[str, int, List[Dict]]:
        """
        Extract text content from PDF using PyMuPDF.
        Handles native text extraction for digital PDFs.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, page_count, page_metadata_list)
        """
        logger.info(f"[STEP 1] Extracting text from PDF: {file_path}")
        
        try:
            doc = fitz.open(file_path)
            full_text = ""
            page_metadata = []
            page_count = len(doc)
            
            for page_num, page in enumerate(doc, start=1):
                # Extract text from page
                page_text = page.get_text()
                char_count = len(page_text.strip())
                
                # Track which pages need OCR
                needs_ocr = char_count < 50
                
                # Add page marker for context
                full_text += f"\n\n{'='*50}\n"
                full_text += f"PAGE {page_num}\n"
                full_text += f"{'='*50}\n\n"
                full_text += page_text
                
                # Store page metadata
                page_metadata.append({
                    "page_number": page_num,
                    "char_count": char_count,
                    "needs_ocr": needs_ocr,
                    "has_native_text": char_count >= 50
                })
                
                logger.debug(f"  Page {page_num}: {char_count} characters extracted")
            
            doc.close()
            logger.info(f"✓ [STEP 1] Extracted text from {page_count} pages")
            return full_text, page_count, page_metadata
            
        except Exception as e:
            logger.error(f"✗ [STEP 1] Error extracting text from PDF: {str(e)}")
            raise
    
    
    # ============================================================================
    # STEP 2: IMPLEMENT PDF TO IMAGE CONVERSION
    # ============================================================================
    
    def convert_pdf_to_images(self, file_path: str, dpi: int = 300) -> List[Dict]:
        """
        Convert PDF pages to high-quality images for OCR processing.
        Uses pdf2image library for accurate conversion.
        
        Args:
            file_path: Path to PDF file
            dpi: Resolution for conversion (default: 300 for good OCR quality)
            
        Returns:
            List of dictionaries with page image information
        """
        logger.info(f"[STEP 2] Converting PDF pages to images: {file_path}")
        
        page_images = []
        
        try:
            pdf_name = Path(file_path).stem
            
            # Convert PDF to images
            images = convert_from_path(file_path, dpi=dpi)
            
            for page_num, image in enumerate(images, start=1):
                # Save page image
                image_filename = f"{pdf_name}_page_{page_num}.png"
                image_path = os.path.join(self.image_dir, image_filename)
                image.save(image_path, "PNG")
                
                page_images.append({
                    "page_number": page_num,
                    "image_path": image_path,
                    "width": image.width,
                    "height": image.height,
                    "dpi": dpi
                })
                
                logger.debug(f"  Page {page_num} → {image_filename}")
            
            logger.info(f"✓ [STEP 2] Converted {len(images)} pages to images")
            return page_images
            
        except Exception as e:
            logger.error(f"✗ [STEP 2] Error converting PDF to images: {str(e)}")
            return []
    
    
    # ============================================================================
    # STEP 3: BUILD OCR MODULE
    # ============================================================================
    
    def enhance_image_for_ocr(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Enhance image quality to improve OCR accuracy.
        Applies denoising, contrast enhancement, and deskewing.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Enhanced image as numpy array
        """
        logger.debug("[STEP 3] Enhancing image for OCR")
        
        try:
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                img = np.array(image)
            else:
                img = image.copy()
            
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            # Apply denoising to remove artifacts
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Apply adaptive thresholding for better text detection
            binary = cv2.adaptiveThreshold(
                enhanced, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Auto-deskew if image is rotated
            coords = np.column_stack(np.where(binary > 0))
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = 90 + angle
                if abs(angle) > 0.5:  # Only deskew if angle is significant
                    (h, w) = binary.shape
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    binary = cv2.warpAffine(
                        binary, M, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE
                    )
            
            return binary
            
        except Exception as e:
            logger.error(f"✗ [STEP 3] Error enhancing image: {str(e)}")
            return np.array(image) if isinstance(image, Image.Image) else image
    
    def perform_ocr(self, image_path: str, enhance: bool = True) -> Tuple[str, Dict]:
        """
        Perform Optical Character Recognition on image.
        Extracts text and provides confidence metrics.
        
        Args:
            image_path: Path to image file
            enhance: Whether to enhance image before OCR
            
        Returns:
            Tuple of (extracted_text, ocr_metadata)
        """
        logger.info(f"[STEP 3] Performing OCR on: {Path(image_path).name}")
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Enhance if enabled
            if enhance and self.enhance_images:
                enhanced = self.enhance_image_for_ocr(image)
                
                # Save enhanced version
                enhanced_filename = f"enhanced_{Path(image_path).name}"
                enhanced_path = os.path.join(self.enhanced_dir, enhanced_filename)
                cv2.imwrite(enhanced_path, enhanced)
                
                # Perform OCR on enhanced image
                text = pytesseract.image_to_string(enhanced)
            else:
                # Perform OCR on original image
                text = pytesseract.image_to_string(image)
            
            # Get confidence score
            try:
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                confidences = [int(c) for c in data['conf'] if c != '-1']
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            except:
                avg_confidence = 0
            
            metadata = {
                "text_length": len(text.strip()),
                "word_count": len(text.split()),
                "ocr_confidence": round(avg_confidence, 2),
                "enhanced": enhance and self.enhance_images
            }
            
            logger.info(f"✓ [STEP 3] OCR completed with {avg_confidence:.2f}% confidence")
            return text, metadata
            
        except Exception as e:
            logger.error(f"✗ [STEP 3] OCR error: {str(e)}")
            return "", {}
    
    def apply_ocr_to_pdf_pages(
        self, 
        file_path: str, 
        page_metadata: List[Dict]
    ) -> Dict[int, str]:
        """
        Apply OCR to PDF pages that need it (scanned or low text content).
        
        Args:
            file_path: Path to PDF file
            page_metadata: List of page metadata from text extraction
            
        Returns:
            Dictionary mapping page numbers to OCR text
        """
        logger.info(f"[STEP 3] Applying OCR to pages that need it")
        
        ocr_results = {}
        
        # Convert pages to images first
        page_images = self.convert_pdf_to_images(file_path)
        
        # Apply OCR to pages that need it
        for page_info, img_info in zip(page_metadata, page_images):
            page_num = page_info["page_number"]
            
            if page_info.get("needs_ocr", False):
                logger.info(f"  Applying OCR to page {page_num}")
                text, ocr_meta = self.perform_ocr(img_info["image_path"])
                ocr_results[page_num] = {
                    "text": text,
                    "metadata": ocr_meta
                }
        
        logger.info(f"✓ [STEP 3] OCR applied to {len(ocr_results)} pages")
        return ocr_results
    
    
    # ============================================================================
    # STEP 4: IMPLEMENT IMAGE EXTRACTION
    # ============================================================================
    
    def extract_images_from_pdf(self, file_path: str) -> List[Dict]:
        """
        Extract embedded images from PDF pages.
        Saves images and optionally performs OCR on them.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of extracted image metadata
        """
        logger.info(f"[STEP 4] Extracting embedded images from PDF")
        
        images_info = []
        
        try:
            doc = fitz.open(file_path)
            pdf_name = Path(file_path).stem
            
            for page_num, page in enumerate(doc, start=1):
                # Get all images on the page
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Generate filename
                        image_filename = f"{pdf_name}_p{page_num}_img{img_index}.{image_ext}"
                        image_path = os.path.join(self.image_dir, image_filename)
                        
                        # Save image
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        # Optionally extract text from image using OCR
                        img_text = ""
                        ocr_conf = 0
                        if self.ocr_enabled:
                            try:
                                img_text, ocr_meta = self.perform_ocr(image_path, enhance=False)
                                ocr_conf = ocr_meta.get("ocr_confidence", 0)
                            except:
                                pass
                        
                        images_info.append({
                            "page": page_num,
                            "image_index": img_index,
                            "path": image_path,
                            "filename": image_filename,
                            "extension": image_ext,
                            "width": base_image.get("width", 0),
                            "height": base_image.get("height", 0),
                            "extracted_text": img_text[:500] if img_text else "",
                            "ocr_confidence": ocr_conf
                        })
                        
                        logger.debug(f"  Extracted: {image_filename}")
                        
                    except Exception as e:
                        logger.warning(f"  Failed to extract image {img_index} from page {page_num}: {e}")
                        continue
            
            doc.close()
            logger.info(f"✓ [STEP 4] Extracted {len(images_info)} images from PDF")
            return images_info
            
        except Exception as e:
            logger.error(f"✗ [STEP 4] Error extracting images: {str(e)}")
            return []
    
    def process_standalone_image(self, file_path: str) -> List[Document]:
        """
        Process standalone image files with OCR.
        
        Args:
            file_path: Path to image file
            
        Returns:
            List containing Document object with extracted text
        """
        logger.info(f"[STEP 4] Processing standalone image: {file_path}")
        
        try:
            # Perform OCR
            text, ocr_metadata = self.perform_ocr(file_path, enhance=True)
            
            # Load image for metadata
            image = Image.open(file_path)
            
            doc_metadata = {
                "source": file_path,
                "file_name": Path(file_path).name,
                "file_type": "image",
                "image_size": image.size,
                "image_mode": image.mode,
                **ocr_metadata,
                **self.metadata
            }
            
            content = text.strip() if text.strip() else "[Image with no detectable text]"
            
            document = Document(page_content=content, metadata=doc_metadata)
            logger.info(f"✓ [STEP 4] Processed image successfully")
            
            return [document]
            
        except Exception as e:
            logger.error(f"✗ [STEP 4] Error processing image: {str(e)}")
            raise
    
    
    # ============================================================================
    # STEP 5: CREATE DOCUMENT CHUNKING STRATEGY
    # ============================================================================
    
    def create_smart_chunks(self, text: str, metadata: Dict) -> List[Document]:
        """
        Split text into semantic chunks with overlap for better context retention.
        Uses hierarchical separators for intelligent splitting.
        
        Args:
            text: Text to split
            metadata: Base metadata to include in all chunks
            
        Returns:
            List of Document chunks
        """
        logger.info(f"[STEP 5] Creating document chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        
        try:
            # Configure text splitter with smart separators
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=[
                    "\n\n",      # Paragraph breaks
                    "\n",        # Line breaks
                    ". ",        # Sentences
                    "! ",        # Exclamations
                    "? ",        # Questions
                    "; ",        # Semicolons
                    ", ",        # Commas
                    " ",         # Spaces
                    ""           # Characters
                ],
                length_function=len,
                is_separator_regex=False
            )
            
            # Split text into chunks
            chunks = text_splitter.split_text(text)
            
            # Create Document objects with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **metadata,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                    "has_overlap": self.chunk_overlap > 0
                }
                
                doc = Document(page_content=chunk, metadata=chunk_metadata)
                documents.append(doc)
            
            logger.info(f"✓ [STEP 5] Created {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"✗ [STEP 5] Chunking error: {str(e)}")
            raise
    
    
    # ============================================================================
    # COMPLETE PDF PROCESSING PIPELINE
    # ============================================================================
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Complete PDF processing pipeline combining all steps.
        
        Pipeline:
        1. Extract text from PDF
        2. Convert pages to images (if needed)
        3. Apply OCR to scanned pages
        4. Extract embedded images
        5. Chunk the text
        6. Return Document objects
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects ready for embedding
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"PROCESSING PDF: {Path(file_path).name}")
        logger.info(f"{'='*70}\n")
        
        try:
            # STEP 1: Extract native text
            text, page_count, page_metadata = self.extract_text_from_pdf(file_path)
            
            # STEP 2 & 3: Apply OCR to pages that need it
            ocr_results = {}
            if self.ocr_enabled:
                ocr_results = self.apply_ocr_to_pdf_pages(file_path, page_metadata)
                
                # Merge OCR results into main text
                for page_num, ocr_data in ocr_results.items():
                    ocr_text = ocr_data["text"]
                    # Replace or append OCR text for that page
                    text += f"\n\n[OCR Page {page_num}]\n{ocr_text}"
            
            # STEP 4: Extract images
            images_info = []
            if self.extract_images:
                images_info = self.extract_images_from_pdf(file_path)
            
            # STEP 5: Create chunks
            base_metadata = {
                "source": file_path,
                "file_name": Path(file_path).name,
                "file_type": "pdf",
                "page_count": page_count,
                "has_images": len(images_info) > 0,
                "image_count": len(images_info),
                "ocr_pages": len(ocr_results),
                **self.metadata
            }
            
            documents = self.create_smart_chunks(text, base_metadata)
            
            # Add image info and page metadata to first document
            if documents:
                documents[0].metadata["images_info"] = images_info
                documents[0].metadata["page_metadata"] = page_metadata
            
            logger.info(f"\n✓ PDF processing complete: {len(documents)} documents created\n")
            return documents
            
        except Exception as e:
            logger.error(f"\n✗ PDF processing failed: {str(e)}\n")
            raise
    
    
    # ============================================================================
    # AUDIO PROCESSING (BONUS)
    # ============================================================================
    def process_audio(self, file_path: str) -> List[Document]:
        """
        Process audio files with speech-to-text transcription.
        """
        logger.info(f"[AUDIO] Processing audio file: {file_path}")

        try:
            recognizer = sr.Recognizer()

            # Convert to WAV if needed
            if not file_path.lower().endswith(".wav"):
                audio = AudioSegment.from_file(file_path)
                wav_path = os.path.join(self.audio_dir, f"{Path(file_path).stem}.wav")
                audio.export(wav_path, format="wav")
                file_path = wav_path

            # Transcribe audio
            with sr.AudioFile(file_path) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                except sr.UnknownValueError:
                    text = "[Audio could not be transcribed]"
                except sr.RequestError as e:
                    text = f"[Transcription service error: {e}]"

            # Audio metadata
            audio = AudioSegment.from_wav(file_path)

            doc_metadata = {
                "source": file_path,
                "file_name": Path(file_path).name,
                "file_type": "audio",
                "duration_seconds": len(audio) / 1000.0,
                "sample_rate": audio.frame_rate,
                "channels": audio.channels,
                **self.metadata
            }

            documents = self.create_smart_chunks(text, doc_metadata)

            logger.info("✓ [AUDIO] Transcribed and processed audio")
            return documents

        except Exception as e:
            logger.error(f"✗ [AUDIO] Error: {str(e)}")
            raise

    # ============================================================================
    # STEP 6: TEST DOCUMENT PROCESSOR
    # ============================================================================
    
    def test_processor(self, test_files: List[str]) -> Dict:
        """
        Test the document processor with various file types.
        Provides comprehensive statistics and error reporting.
        
        Args:
            test_files: List of file paths to test
            
        Returns:
            Dictionary with test results and statistics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"TESTING DOCUMENT PROCESSOR")
        logger.info(f"{'='*70}\n")
        
        results = {
            "total_files": len(test_files),
            "successful": 0,
            "failed": 0,
            "total_documents": 0,
            "by_type": {},
            "errors": []
        }
        
        for file_path in test_files:
            try:
                logger.info(f"Testing: {file_path}")
                
                # Process document
                documents = self.process_document(file_path)
                
                # Update statistics
                results["successful"] += 1
                results["total_documents"] += len(documents)
                
                file_type = Path(file_path).suffix.lower()
                if file_type not in results["by_type"]:
                    results["by_type"][file_type] = {"count": 0, "docs": 0}
                results["by_type"][file_type]["count"] += 1
                results["by_type"][file_type]["docs"] += len(documents)
                
                logger.info(f"  ✓ Success: {len(documents)} documents created\n")
                
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({"file": file_path, "error": str(e)})
                logger.error(f"  ✗ Failed: {str(e)}\n")
        
        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info(f"TEST SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total files tested: {results['total_files']}")
        logger.info(f"Successful: {results['successful']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Total documents created: {results['total_documents']}")
        logger.info(f"\nBy file type:")
        for file_type, stats in results["by_type"].items():
            logger.info(f"  {file_type}: {stats['count']} files → {stats['docs']} documents")
        
        if results["errors"]:
            logger.info(f"\nErrors:")
            for error in results["errors"]:
                logger.info(f"  {error['file']}: {error['error']}")
        
        logger.info(f"{'='*70}\n")
        
        return results
    
    
    # ============================================================================
    # MAIN PROCESSING INTERFACE
    # ============================================================================
    
    def process_document(self, file_path: str) -> List[Document]:
        """
        Main entry point for processing any supported document type.
        
        Supported formats:
        - PDF: .pdf
        - Images: .png, .jpg, .jpeg, .tiff, .bmp, .gif
        - Audio: .wav, .mp3, .m4a, .flac, .ogg
        
        Args:
            file_path: Path to document file
            
        Returns:
            List of Document objects ready for embedding
        """
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        
        # Route to appropriate processor
        if file_extension == ".pdf":
            return self.process_pdf(file_path)
        
        elif file_extension in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            return self.process_standalone_image(file_path)
        
        elif file_extension in [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"]:
            if not self.process_audio_enabled:
                raise ValueError("Audio processing is disabled")
            return self.process_audio(file_path)
        
        else:
            raise ValueError(
                f"Unsupported file type: {file_extension}\n"
                f"Supported: .pdf, .png/.jpg/.jpeg, .wav/.mp3"
            )
    
    def process_multiple_documents(
        self, 
        file_paths: List[str],
        skip_errors: bool = True
    ) -> List[Document]:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of file paths to process
            skip_errors: Continue processing if a file fails
            
        Returns:
            Combined list of all Document objects
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"BATCH PROCESSING: {len(file_paths)} files")
        logger.info(f"{'='*70}\n")
        
        all_documents = []
        failed_files = []
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                logger.info(f"[{i}/{len(file_paths)}] Processing: {file_path}")
                documents = self.process_document(file_path)
                all_documents.extend(documents)
                logger.info(f"  ✓ Created {len(documents)} documents\n")
                
            except Exception as e:
                logger.error(f"  ✗ Error: {str(e)}\n")
                failed_files.append(file_path)
                if not skip_errors:
                    raise
        
        # Summary
        success_count = len(file_paths) - len(failed_files)
        logger.info(f"\n{'='*70}")
        logger.info(f"BATCH COMPLETE: {success_count}/{len(file_paths)} successful")
        logger.info(f"Total documents created: {len(all_documents)}")
        if failed_files:
            logger.info(f"Failed files: {failed_files}")
        logger.info(f"{'='*70}\n")
        
        return all_documents


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MULTIMODAL DOCUMENT PREPROCESSOR - DEMO")
    print("=" * 70 + "\n")

    # Initialize preprocessor
    preprocessor = MultimodalPreprocessor(
        chunk_size=1000,
        chunk_overlap=200,
        extract_images=True,
        ocr_enabled=True,
        process_audio_enabled=True,
        enhance_images=True,
        output_dir="preprocessed_data",
        metadata={"project": "multimodal_rag", "version": "1.0"}
    )

    # ------------------------------------------------------------------
    # Define test file paths (portable)
    # ------------------------------------------------------------------
    BASE_DIR = Path(__file__).parent / "testfiles"

    pdf_path = BASE_DIR / "sample.pdf"
    image_path = BASE_DIR / "sample_image.png"
    audio_path = BASE_DIR / "sample_audio.m4a"

    test_files = [str(pdf_path), str(image_path), str(audio_path)]

    # ------------------------------------------------------------------
    # Example 1: Process PDF
    # ------------------------------------------------------------------
    print("\n--- Example 1: Process PDF ---")
    try:
        pdf_docs = preprocessor.process_document(str(pdf_path))
        print(f"✓ Created {len(pdf_docs)} document chunks")
        print(f"First chunk preview:\n{pdf_docs[0].page_content[:200]}...")
        print(f"\nMetadata: {pdf_docs[0].metadata}")
    except FileNotFoundError:
        print("⚠ sample.pdf not found - skipping this example")
    except Exception as e:
        print(f"✗ Error: {e}")

    # ------------------------------------------------------------------
    # Example 2: Process Image
    # ------------------------------------------------------------------
    print("\n--- Example 2: Process Image ---")
    try:
        image_docs = preprocessor.process_document(str(image_path))
        print(f"✓ Created {len(image_docs)} document(s)")
        print(f"Extracted text preview:\n{image_docs[0].page_content[:200]}...")
        print(f"\nMetadata: {image_docs[0].metadata}")
    except FileNotFoundError:
        print("⚠ sample_image.png not found - skipping this example")
    except Exception as e:
        print(f"✗ Error: {e}")

    # ------------------------------------------------------------------
    # Example 3: Process Audio
    # ------------------------------------------------------------------
    print("\n--- Example 3: Process Audio ---")
    try:
        audio_docs = preprocessor.process_document(str(audio_path))
        print(f"✓ Created {len(audio_docs)} document chunk(s)")
        print(f"Transcript preview:\n{audio_docs[0].page_content[:200]}...")
        print(f"\nMetadata: {audio_docs[0].metadata}")
    except FileNotFoundError:
        print("⚠ sample_audio.m4a not found - skipping this example")
    except Exception as e:
        print(f"✗ Error: {e}")

    # ------------------------------------------------------------------
    # Example 4: Batch Processing
    # ------------------------------------------------------------------
    print("\n--- Example 4: Batch Processing ---")
    try:
        all_docs = preprocessor.process_multiple_documents(test_files, skip_errors=True)
        print("\n✓ Batch processing completed")
        print(f"Total documents created: {len(all_docs)}")
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")

    # ------------------------------------------------------------------
    # Example 5: Processor Test Suite
    # ------------------------------------------------------------------
    print("\n--- Example 5: Processor Test Suite ---")
    try:
        test_results = preprocessor.test_processor(test_files)
        print("\nTest Results Summary:")
        print(test_results)
    except Exception as e:
        print(f"✗ Testing failed: {e}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
