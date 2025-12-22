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
    Comprehensive preprocessor for documents, images, and audio files.
    Converts all modalities into LangChain Documents with rich metadata.
    
    Attributes:
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between consecutive chunks
        extract_images: Whether to extract images from PDFs
        ocr_enabled: Whether to use OCR for scanned documents
        process_audio: Whether to process audio files
        enhance_images: Whether to apply image enhancement
        output_dir: Base directory for outputs
    """
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    extract_images: bool = True
    ocr_enabled: bool = True
    process_audio: bool = True
    enhance_images: bool = True
    output_dir: str = "preprocessed_outputs"
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize output directories."""
        self.image_dir = os.path.join(self.output_dir, "images")
        self.audio_dir = os.path.join(self.output_dir, "audio")
        self.enhanced_dir = os.path.join(self.output_dir, "enhanced")
        
        for directory in [self.image_dir, self.audio_dir, self.enhanced_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized preprocessor with output dir: {self.output_dir}")
    
    # ==================== IMAGE PREPROCESSING ====================
    
    def enhance_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Enhance image quality for better OCR results.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Enhanced image as numpy array
        """
        try:
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                img = np.array(image)
            else:
                img = image.copy()
            
            # Convert to grayscale if color
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # Increase contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Apply adaptive thresholding for better text detection
            binary = cv2.adaptiveThreshold(
                enhanced, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Deskew if needed
            coords = np.column_stack(np.where(binary > 0))
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = 90 + angle
                if abs(angle) > 0.5:  # Only deskew if needed
                    (h, w) = binary.shape
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    binary = cv2.warpAffine(
                        binary, M, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE
                    )
            
            logger.debug("Image enhancement completed")
            return binary
            
        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            return np.array(image) if isinstance(image, Image.Image) else image
    
    def extract_text_from_image(self, image_path: str) -> Tuple[str, Dict]:
        """
        Extract text from image using OCR with optional enhancement.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (extracted_text, image_metadata)
        """
        try:
            # Load image
            image = Image.open(image_path)
            original_size = image.size
            
            # Enhance if enabled
            if self.enhance_images:
                enhanced = self.enhance_image(image)
                
                # Save enhanced version
                enhanced_filename = f"enhanced_{Path(image_path).name}"
                enhanced_path = os.path.join(self.enhanced_dir, enhanced_filename)
                cv2.imwrite(enhanced_path, enhanced)
                
                # Use enhanced image for OCR
                text = pytesseract.image_to_string(enhanced)
            else:
                text = pytesseract.image_to_string(image)
            
            # Get OCR confidence
            try:
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            except:
                avg_confidence = 0
            
            metadata = {
                "original_size": original_size,
                "image_mode": image.mode,
                "text_length": len(text.strip()),
                "ocr_confidence": round(avg_confidence, 2),
                "enhanced": self.enhance_images
            }
            
            logger.info(f"Extracted text from image with {avg_confidence:.2f}% confidence")
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return "", {}
    
    def process_image(self, file_path: str) -> List[Document]:
        """
        Process standalone image files.
        
        Args:
            file_path: Path to image file
            
        Returns:
            List containing Document object
        """
        try:
            text, img_metadata = self.extract_text_from_image(file_path)
            
            doc_metadata = {
                "source": file_path,
                "file_name": Path(file_path).name,
                "file_type": "image",
                **img_metadata,
                **self.metadata
            }
            
            content = text.strip() if text.strip() else "[Image with no detectable text]"
            
            document = Document(page_content=content, metadata=doc_metadata)
            logger.info(f"Processed image: {file_path}")
            
            return [document]
            
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {str(e)}")
            raise
    
    # ==================== PDF PREPROCESSING ====================
    
    def extract_text_from_pdf(self, file_path: str) -> Tuple[str, int, List[Dict]]:
        """
        Extract text from PDF with page-level metadata.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (text, page_count, page_metadata)
        """
        try:
            doc = fitz.open(file_path)
            text = ""
            page_metadata = []
            page_count = len(doc)
            
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text()
                char_count = len(page_text.strip())
                
                # If page has very little text, try OCR
                if self.ocr_enabled and char_count < 50:
                    logger.info(f"Page {page_num}: Minimal text detected, applying OCR...")
                    
                    # Convert page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Apply enhancement and OCR
                    if self.enhance_images:
                        enhanced = self.enhance_image(img)
                        page_text = pytesseract.image_to_string(enhanced)
                    else:
                        page_text = pytesseract.image_to_string(img)
                    
                    char_count = len(page_text.strip())
                
                text += f"\n\n=== Page {page_num} ===\n{page_text}"
                
                page_metadata.append({
                    "page_number": page_num,
                    "char_count": char_count,
                    "ocr_applied": char_count < 50
                })
            
            doc.close()
            logger.info(f"Extracted text from {page_count} pages")
            return text, page_count, page_metadata
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def extract_images_from_pdf(self, file_path: str) -> List[Dict]:
        """
        Extract and save images from PDF pages.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of image metadata dictionaries
        """
        images_info = []
        
        try:
            doc = fitz.open(file_path)
            pdf_name = Path(file_path).stem
            
            for page_num, page in enumerate(doc, start=1):
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Save image
                        image_filename = f"{pdf_name}_p{page_num}_i{img_index}.{image_ext}"
                        image_path = os.path.join(self.image_dir, image_filename)
                        
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        # Try to extract text from image if OCR enabled
                        img_text = ""
                        if self.ocr_enabled:
                            try:
                                img_text, _ = self.extract_text_from_image(image_path)
                            except:
                                pass
                        
                        images_info.append({
                            "page": page_num,
                            "image_index": img_index,
                            "path": image_path,
                            "extension": image_ext,
                            "width": base_image.get("width", 0),
                            "height": base_image.get("height", 0),
                            "extracted_text": img_text[:200] if img_text else ""
                        })
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                        continue
            
            doc.close()
            logger.info(f"Extracted {len(images_info)} images from PDF")
            return images_info
            
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {str(e)}")
            return []
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Process PDF with text extraction, OCR, and image extraction.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects
        """
        try:
            # Extract text and metadata
            text, page_count, page_metadata = self.extract_text_from_pdf(file_path)
            
            # Extract images if enabled
            images_info = []
            if self.extract_images:
                images_info = self.extract_images_from_pdf(file_path)
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
                length_function=len
            )
            
            chunks = text_splitter.split_text(text)
            
            # Create documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    "source": file_path,
                    "file_name": Path(file_path).name,
                    "file_type": "pdf",
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "page_count": page_count,
                    "has_images": len(images_info) > 0,
                    "image_count": len(images_info),
                    **self.metadata
                }
                
                documents.append(Document(page_content=chunk, metadata=doc_metadata))
            
            # Store image info in first document
            if images_info and documents:
                documents[0].metadata["images_info"] = images_info
                documents[0].metadata["page_metadata"] = page_metadata
            
            logger.info(f"Created {len(documents)} chunks from PDF")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
    
    # ==================== AUDIO PREPROCESSING ====================
    
    def convert_audio_format(self, file_path: str, target_format: str = "wav") -> str:
        """
        Convert audio to target format.
        
        Args:
            file_path: Path to audio file
            target_format: Target format (wav, mp3, etc.)
            
        Returns:
            Path to converted file
        """
        try:
            audio = AudioSegment.from_file(file_path)
            
            output_filename = f"{Path(file_path).stem}.{target_format}"
            output_path = os.path.join(self.audio_dir, output_filename)
            
            audio.export(output_path, format=target_format)
            logger.info(f"Converted audio to {target_format}: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting audio format: {str(e)}")
            raise
    
    def enhance_audio(self, file_path: str) -> str:
        """
        Enhance audio quality for better transcription.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Path to enhanced audio file
        """
        try:
            # Load audio
            y, sr_rate = librosa.load(file_path, sr=16000)
            
            # Noise reduction using spectral gating
            # Simple approach: reduce very quiet parts
            threshold = np.percentile(np.abs(y), 20)
            y_reduced = np.where(np.abs(y) > threshold, y, y * 0.1)
            
            # Normalize audio
            y_normalized = librosa.util.normalize(y_reduced)
            
            # Save enhanced audio
            enhanced_filename = f"enhanced_{Path(file_path).name}"
            enhanced_path = os.path.join(self.audio_dir, enhanced_filename)
            sf.write(enhanced_path, y_normalized, sr_rate)
            
            logger.info(f"Enhanced audio saved: {enhanced_path}")
            return enhanced_path
            
        except Exception as e:
            logger.error(f"Error enhancing audio: {str(e)}")
            return file_path  # Return original if enhancement fails
    
    def transcribe_audio(self, file_path: str) -> Tuple[str, Dict]:
        """
        Transcribe audio to text using speech recognition.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (transcription, audio_metadata)
        """
        try:
            recognizer = sr.Recognizer()
            
            # Ensure audio is in WAV format
            if not file_path.lower().endswith('.wav'):
                file_path = self.convert_audio_format(file_path, "wav")
            
            # Load audio
            audio = AudioSegment.from_wav(file_path)
            duration = len(audio) / 1000.0  # Duration in seconds
            
            # Split on silence for long audio files
            if duration > 60:  # If longer than 1 minute
                logger.info("Long audio detected, splitting on silence...")
                chunks = split_on_silence(
                    audio,
                    min_silence_len=500,
                    silence_thresh=audio.dBFS - 14,
                    keep_silence=500
                )
            else:
                chunks = [audio]
            
            # Transcribe each chunk
            full_transcription = []
            for i, chunk in enumerate(chunks):
                # Export chunk to temporary file
                chunk_path = os.path.join(self.audio_dir, f"temp_chunk_{i}.wav")
                chunk.export(chunk_path, format="wav")
                
                # Transcribe
                with sr.AudioFile(chunk_path) as source:
                    audio_data = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio_data)
                        full_transcription.append(text)
                    except sr.UnknownValueError:
                        logger.warning(f"Could not understand audio chunk {i}")
                    except sr.RequestError as e:
                        logger.error(f"API error: {e}")
                
                # Clean up temp file
                os.remove(chunk_path)
            
            transcription = " ".join(full_transcription)
            
            metadata = {
                "duration_seconds": duration,
                "sample_rate": audio.frame_rate,
                "channels": audio.channels,
                "chunk_count": len(chunks),
                "transcription_length": len(transcription)
            }
            
            logger.info(f"Transcribed {duration:.2f}s of audio")
            return transcription, metadata
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return "", {}
    
    def process_audio(self, file_path: str) -> List[Document]:
        """
        Process audio file with transcription.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            List containing Document object
        """
        try:
            # Enhance audio if enabled
            if self.enhance_images:  # Using same flag for audio enhancement
                enhanced_path = self.enhance_audio(file_path)
                transcription, audio_metadata = self.transcribe_audio(enhanced_path)
            else:
                transcription, audio_metadata = self.transcribe_audio(file_path)
            
            doc_metadata = {
                "source": file_path,
                "file_name": Path(file_path).name,
                "file_type": "audio",
                **audio_metadata,
                **self.metadata
            }
            
            content = transcription if transcription else "[Audio file with no transcribable speech]"
            
            # If transcription is very long, split into chunks
            if len(transcription) > self.chunk_size:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                chunks = text_splitter.split_text(transcription)
                
                documents = []
                for i, chunk in enumerate(chunks):
                    chunk_metadata = doc_metadata.copy()
                    chunk_metadata["chunk_id"] = i
                    chunk_metadata["total_chunks"] = len(chunks)
                    documents.append(Document(page_content=chunk, metadata=chunk_metadata))
                
                logger.info(f"Created {len(documents)} chunks from audio transcription")
                return documents
            else:
                document = Document(page_content=content, metadata=doc_metadata)
                logger.info(f"Processed audio: {file_path}")
                return [document]
            
        except Exception as e:
            logger.error(f"Error processing audio {file_path}: {str(e)}")
            raise
    
    # ==================== MAIN PROCESSING INTERFACE ====================
    
    def process_document(self, file_path: str) -> List[Document]:
        """
        Process any supported document type.
        
        Args:
            file_path: Path to document
            
        Returns:
            List of Document objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        logger.info(f"Processing {file_extension} file: {file_path}")
        
        # Route to appropriate processor
        if file_extension == ".pdf":
            return self.process_pdf(file_path)
        
        elif file_extension in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            return self.process_image(file_path)
        
        elif file_extension in [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"]:
            if not self.process_audio:
                raise ValueError("Audio processing is disabled. Set process_audio=True")
            return self.process_audio(file_path)
        
        else:
            raise ValueError(
                f"Unsupported file type: {file_extension}. "
                f"Supported: PDF, Images (PNG/JPG/etc), Audio (WAV/MP3/etc)"
            )
    
    def process_multiple_documents(
        self, 
        file_paths: List[str],
        skip_errors: bool = True
    ) -> List[Document]:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of file paths
            skip_errors: Whether to skip files that fail processing
            
        Returns:
            Combined list of all Documents
        """
        all_documents = []
        failed_files = []
        
        for file_path in file_paths:
            try:
                documents = self.process_document(file_path)
                all_documents.extend(documents)
                logger.info(f"✓ Successfully processed: {file_path}")
            except Exception as e:
                logger.error(f"✗ Failed to process {file_path}: {str(e)}")
                failed_files.append(file_path)
                if not skip_errors:
                    raise
        
        logger.info(
            f"Batch processing complete: "
            f"{len(file_paths) - len(failed_files)}/{len(file_paths)} succeeded, "
            f"{len(all_documents)} total documents created"
        )
        
        if failed_files:
            logger.warning(f"Failed files: {failed_files}")
        
        return all_documents
    
    def get_statistics(self, documents: List[Document]) -> Dict:
        """
        Get statistics about processed documents.
        
        Args:
            documents: List of processed documents
            
        Returns:
            Dictionary with statistics
        """
        if not documents:
            return {"total_documents": 0}
        
        file_types = {}
        total_chars = 0
        sources = set()
        
        for doc in documents:
            file_type = doc.metadata.get("file_type", "unknown")
            file_types[file_type] = file_types.get(file_type, 0) + 1
            total_chars += len(doc.page_content)
            sources.add(doc.metadata.get("source", "unknown"))
        
        stats = {
            "total_documents": len(documents),
            "unique_sources": len(sources),
            "file_types": file_types,
            "total_characters": total_chars,
            "avg_chunk_size": total_chars // len(documents) if documents else 0
        }
        
        return stats


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = MultimodalPreprocessor(
        chunk_size=1000,
        chunk_overlap=200,
        extract_images=True,
        ocr_enabled=True,
        process_audio=True,
        enhance_images=True,
        output_dir="preprocessed_data",
        metadata={"project": "multimodal_rag", "version": "1.0"}
    )
    
    # Example 1: Process single PDF
    print("\n=== Processing PDF ===")
    try:
        pdf_docs = preprocessor.process_document("sample.pdf")
        print(f"Created {len(pdf_docs)} chunks from PDF")
        print(f"First chunk: {pdf_docs[0].page_content[:200]}...")
        print(f"Metadata: {pdf_docs[0].metadata}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Process image
    print("\n=== Processing Image ===")
    try:
        img_docs = preprocessor.process_document("sample.jpg")
        print(f"Extracted text: {img_docs[0].page_content[:200]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Process audio
    print("\n=== Processing Audio ===")
    try:
        audio_docs = preprocessor.process_document("sample.mp3")
        print(f"Transcription: {audio_docs[0].page_content[:200]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 4: Batch processing
    print("\n=== Batch Processing ===")
    file_list = [
        "doc1.pdf",
        "image1.png",
        "audio1.wav",
        "doc2.pdf"
    ]
    
    all_docs = preprocessor.process_multiple_documents(file_list, skip_errors=True)
    
    # Get statistics
    stats = preprocessor.get_statistics(all_docs)
    print(f"\nProcessing Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")