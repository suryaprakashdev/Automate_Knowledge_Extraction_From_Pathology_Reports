#!/usr/bin/env python3
"""
Biomedical NLP Pipeline - Using Microsoft BiomedBERT
"""

from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm

# Microsoft BiomedBERT
from sentence_transformers import SentenceTransformer

# spaCy for NER
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("spaCy not available. Install: pip install spacy scispacy")


class BiomedBERTPipeline:
    """
    Pipeline using Microsoft BiomedBERT embeddings + spaCy NER
    """
    
    def __init__(self, biomedbert_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"):
        """
        Initialize with Microsoft BiomedBERT
        
        Args:
            biomedbert_model: HuggingFace model name
        """
        print(f" Loading Microsoft BiomedBERT: {biomedbert_model}")
        print("   (First run downloads ~400MB, then cached)")
        
        self.embedder = SentenceTransformer(biomedbert_model)
        
        print(f" BiomedBERT loaded (embedding dim: {self.embedder.get_sentence_embedding_dimension()})")
        
        # Load spaCy medical model
        if SPACY_AVAILABLE:
            print(" Loading medical spaCy model...")
            try:
                # Try medical model first
                self.nlp = spacy.load("en_core_sci_md")
                print("Medical spaCy model (en_core_sci_md) loaded")
            except:
                try:
                    # Fallback to general model
                    self.nlp = spacy.load("en_core_web_sm")
                    print(" General spaCy model (en_core_web_sm) loaded")
                except:
                    print("  No spaCy model found. Running without NER.")
                    self.nlp = None
        else:
            self.nlp = None
    
    def process_text(self, text: str) -> Dict:
        """
        Process text with BiomedBERT embeddings + NER
        
        Args:
            text: Input text
        
        Returns:
            Dict with embeddings and entities
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "embeddings": None,
            "entities": []
        }
        
        # Generate embeddings with BiomedBERT
        embedding = self.embedder.encode(text, convert_to_numpy=True)
        result["embeddings"] = embedding.tolist()
        
        # Extract entities with spaCy
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                result["entities"].append({
                    "text": ent.text,
                    "type": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        return result
    
    def process_directory(self, input_dir: str, output_dir: str, save_embeddings: bool = True):
        """
        Process all text files in directory
        
        Args:
            input_dir: Directory with text files
            output_dir: Output directory
            save_embeddings: Whether to save embeddings (can be large!)
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = list(input_dir.glob("*.txt"))
        
        if not files:
            print(f" No .txt files found in {input_dir}")
            return []
        
        print(f"\n Found {len(files)} text files")
        print(f" Processing with Microsoft BiomedBERT...\n")
        
        all_results = []
        success_count = 0
        failed_count = 0
        
        for txt_file in tqdm(files, desc="Processing files"):
            try:
                text = txt_file.read_text(encoding="utf-8")
                
                result = self.process_text(text)
                result["filename"] = txt_file.stem
                
                # Don't save embeddings to JSON (too large)
                # Save them separately if needed
                if save_embeddings:
                    # Save embeddings as numpy
                    import numpy as np
                    emb_file = output_dir / f"{txt_file.stem}_embedding.npy"
                    np.save(emb_file, result["embeddings"])
                
                # Save entities and metadata (without embeddings)
                output_data = {
                    "filename": result["filename"],
                    "timestamp": result["timestamp"],
                    "entities": result["entities"],
                    "entity_count": len(result["entities"]),
                    "has_embedding": save_embeddings
                }
                
                out_file = output_dir / f"{txt_file.stem}_nlp.json"
                with open(out_file, "w") as f:
                    json.dump(output_data, f, indent=2)
                
                all_results.append(output_data)
                success_count += 1
                
                if success_count % 100 == 0:
                    print(f"\n Progress: {success_count}/{len(files)} files")
                
            except Exception as e:
                failed_count += 1
                print(f"\n  FAILED | {txt_file.name} | {e}")
        
        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Total files   : {len(files)}")
        print(f"Successful    : {success_count}")
        print(f"Failed        : {failed_count}")
        print(f"Success rate  : {(success_count/len(files)*100):.1f}%")
        print(f"{'='*70}")
        
        self._save_summary(all_results, output_dir)
        return all_results
    
    def _save_summary(self, results: List[Dict], output_dir: Path):
        """Save processing summary"""
        summary = {
            "total_files": len(results),
            "total_entities": sum(len(r["entities"]) for r in results),
            "timestamp": datetime.now().isoformat(),
            "entity_types": {},
            "model": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        }
        
        for r in results:
            for e in r["entities"]:
                summary["entity_types"][e["type"]] = \
                    summary["entity_types"].get(e["type"], 0) + 1
        
        with open(output_dir / "processing_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "=" * 70)
        print("BIOMEDBERT PROCESSING SUMMARY")
        print("=" * 70)
        print(f"Model         : Microsoft BiomedBERT")
        print(f"Total entities: {summary['total_entities']:,}")
        print(f"\nTop Entity Types:")
        sorted_types = sorted(summary["entity_types"].items(), 
                            key=lambda x: x[1], reverse=True)
        for etype, count in sorted_types[:15]:
            print(f"  {etype:20s} : {count:6,}")
        print("=" * 70)


def main():
    """Main function"""
    print("= " * 20)
    print("MICROSOFT BIOMEDBERT PIPELINE")
    print("= " * 20)
    
    # CONFIGURE PATHS
    input_dir = "/usr/users/3d_dimension_est/selva_sur/RAG/output/text"
    output_dir = "/usr/users/3d_dimension_est/selva_sur/RAG/output/biomedbert_output"
    
    print(f"\nConfiguration:")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")
    print(f"\nModel:")
    print(f"  • Microsoft BiomedBERT (HuggingFace)")
    print(f"  • spaCy medical NER")
    print("="*70)
    
    try:
        pipeline = BiomedBERTPipeline()
        
        results = pipeline.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            save_embeddings=True  # Set False to save space
        )
        
        print(f"\n COMPLETE: {len(results)} files processed")
        print(f"Results: {output_dir}")
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
