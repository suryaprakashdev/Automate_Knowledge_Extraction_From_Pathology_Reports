#!/usr/bin/env python3
"""
BiomedBERT Embeddings → FAISS Vector Database
Creates searchable vector database from BiomedBERT embeddings
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import pickle
from datetime import datetime

# FAISS
import faiss


class BiomedBERTToVectorDB:
    """
    Create FAISS vector database from BiomedBERT embeddings
    """
    
    def __init__(self, 
                 faiss_index_type: str = "hnsw",
                 output_dir: str = "biomedbert_vector_db"):
        """
        Initialize the pipeline
        
        Args:
            faiss_index_type: Type of FAISS index (flat, ivf, hnsw)
            output_dir: Directory to save FAISS index and metadata
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.faiss_index_type = faiss_index_type
        self.embedding_dim = 768  # BiomedBERT dimension
        
        # Initialize FAISS index
        self.index = self._create_faiss_index()
        
        # Storage for metadata
        self.chunks = []
        self.chunk_id_to_idx = {}
        
        self.stats = {
            "total_files": 0,
            "total_chunks": 0,
            "total_entities": 0,
            "entity_types": {},
            "files_processed": []
        }
    
    def _create_faiss_index(self) -> faiss.Index:
        """Create FAISS index based on type"""
        
        if self.faiss_index_type == "flat":
            # Exact search (slower but accurate)
            index = faiss.IndexFlatL2(self.embedding_dim)
            print(f" Created FAISS Flat index (exact search)")
            
        elif self.faiss_index_type == "ivf":
            # Inverted file index (faster approximate search)
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
            print(f" Created FAISS IVF index (approximate search)")
            
        elif self.faiss_index_type == "hnsw":
            # Hierarchical Navigable Small World (best balance)
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            print(f" Created FAISS HNSW index (balanced)")
            
        else:
            raise ValueError(f"Unknown index type: {self.faiss_index_type}")
        
        return index
    
    def load_embedding_and_metadata(self, base_path: Path) -> tuple:
        """
        Load embedding (.npy) and metadata (.json) for a file
        
        Args:
            base_path: Path without extension (e.g., "output/file1")
        
        Returns:
            (embedding, metadata) tuple
        """
        # Load embedding
        emb_file = base_path.parent / f"{base_path.stem}_embedding.npy"
        if not emb_file.exists():
            return None, None
        
        embedding = np.load(emb_file)
        
        # Load metadata
        json_file = base_path.parent / f"{base_path.stem}_nlp.json"
        if not json_file.exists():
            return embedding, {}
        
        with open(json_file, 'r') as f:
            metadata = json.load(f)
        
        return embedding, metadata
    
    def create_chunks_from_text(self, 
                                text: str, 
                                filename: str,
                                entities: List[Dict],
                                chunk_size: int = 512) -> List[Dict]:
        """
        Create text chunks with entity metadata
        
        Args:
            text: Original text
            filename: Source filename
            entities: List of entities from NER
            chunk_size: Characters per chunk
        
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        # Split into sentences
        sentences = text.split('. ')
        
        current_chunk_text = []
        current_chunk_entities = []
        chunk_id = 0
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_with_period = sentence + '. '
            sentence_length = len(sentence_with_period)
            
            # Find entities in this sentence
            sentence_entities = []
            for entity in entities:
                if entity['text'].lower() in sentence.lower():
                    sentence_entities.append(entity)
            
            # Check if adding this sentence exceeds chunk size
            if current_length + sentence_length > chunk_size and current_chunk_text:
                # Save current chunk
                chunk_text = ''.join(current_chunk_text)
                
                chunk = {
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'filename': filename,
                    'entities': current_chunk_entities,
                    'entity_count': len(current_chunk_entities),
                    'entity_types': list(set([e['type'] for e in current_chunk_entities]))
                }
                
                chunks.append(chunk)
                
                # Start new chunk (with overlap - keep last sentence)
                current_chunk_text = [sentence_with_period]
                current_chunk_entities = sentence_entities.copy()
                current_length = sentence_length
                chunk_id += 1
            else:
                # Add to current chunk
                current_chunk_text.append(sentence_with_period)
                current_chunk_entities.extend(sentence_entities)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk_text:
            chunk_text = ''.join(current_chunk_text)
            chunk = {
                'chunk_id': chunk_id,
                'text': chunk_text,
                'filename': filename,
                'entities': current_chunk_entities,
                'entity_count': len(current_chunk_entities),
                'entity_types': list(set([e['type'] for e in current_chunk_entities]))
            }
            chunks.append(chunk)
        
        return chunks
    
    def process_file(self, 
                    embedding_file: Path,
                    original_text_dir: Path = None) -> List[tuple]:
        """
        Process a single file's embedding and metadata
        
        Args:
            embedding_file: Path to .npy embedding file
            original_text_dir: Optional directory with original .txt files
        
        Returns:
            List of (embedding, chunk_metadata) tuples
        """
        # Get base name (without _embedding.npy)
        base_name = embedding_file.stem.replace('_embedding', '')
        
        # Load embedding and metadata
        base_path = embedding_file.parent / base_name
        embedding, metadata = self.load_embedding_and_metadata(base_path)
        
        if embedding is None:
            return []
        
        filename = metadata.get('filename', base_name)
        entities = metadata.get('entities', [])
        
        # Update stats
        self.stats['total_entities'] += len(entities)
        for entity in entities:
            etype = entity.get('type', 'UNKNOWN')
            self.stats['entity_types'][etype] = self.stats['entity_types'].get(etype, 0) + 1
        
        # Option 1: If we have original text, create multiple chunks
        if original_text_dir:
            txt_file = original_text_dir / f"{base_name}.txt"
            if txt_file.exists():
                text = txt_file.read_text(encoding='utf-8')
                # Remove header if present
                if text.startswith('# GDC Pathology Report'):
                    lines = text.split('\n')
                    text = '\n'.join([l for l in lines if not l.startswith('#')])
                
                chunks = self.create_chunks_from_text(text, filename, entities)
                
                # Each chunk gets the same embedding (for now)
                # In production, you'd generate separate embeddings per chunk
                return [(embedding, chunk) for chunk in chunks]
        
        # Option 2: Use single embedding for whole document
        chunk = {
            'chunk_id': 0,
            'text': f"Document: {filename}",
            'filename': filename,
            'entities': entities,
            'entity_count': len(entities),
            'entity_types': list(set([e['type'] for e in entities]))
        }
        
        return [(embedding, chunk)]
    
    def add_to_faiss(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Add embeddings and chunks to FAISS index
        
        Args:
            embeddings: numpy array of embeddings (N x 768)
            chunks: List of chunk metadata dicts
        """
        # Train index if needed (for IVF)
        if self.faiss_index_type == "ivf" and not self.index.is_trained:
            print(" Training IVF index...")
            self.index.train(embeddings.astype('float32'))
        
        # Add to index
        start_idx = self.index.ntotal
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        for i, chunk in enumerate(chunks):
            chunk_id = f"{chunk['filename']}_{chunk['chunk_id']}"
            self.chunk_id_to_idx[chunk_id] = start_idx + i
            self.chunks.append(chunk)
        
        return start_idx
    
    def process_directory(self, 
                         biomedbert_output_dir: str,
                         original_text_dir: str = None):
        """
        Process all BiomedBERT embeddings in a directory
        
        Args:
            biomedbert_output_dir: Directory containing *_embedding.npy and *_nlp.json files
            original_text_dir: Optional directory with original .txt files for better chunking
        """
        output_dir = Path(biomedbert_output_dir)
        embedding_files = sorted(output_dir.glob("*_embedding.npy"))
        
        if not embedding_files:
            print(f"No *_embedding.npy files found in {biomedbert_output_dir}")
            return
        
        print(f"\n Found {len(embedding_files)} embedding files")
        print(f" Creating FAISS vector database...\n")
        
        # Optional: Load original texts for better chunking
        original_text_path = Path(original_text_dir) if original_text_dir else None
        
        all_embeddings = []
        all_chunks = []
        
        # Process each file
        for emb_file in tqdm(embedding_files, desc="Processing embeddings"):
            self.stats['total_files'] += 1
            
            try:
                results = self.process_file(emb_file, original_text_path)
                
                for embedding, chunk in results:
                    all_embeddings.append(embedding)
                    all_chunks.append(chunk)
                
                self.stats['files_processed'].append(emb_file.stem.replace('_embedding', ''))
                
            except Exception as e:
                print(f"\n  Error processing {emb_file.name}: {e}")
        
        if not all_embeddings:
            print(" No embeddings processed. Check your files.")
            return
        
        self.stats['total_chunks'] = len(all_chunks)
        
        print(f"\n Loaded {len(all_embeddings)} embeddings")
        print(f" Created {len(all_chunks)} chunks")
        
        # Stack embeddings
        print(f"\n Adding to FAISS index...")
        embeddings_matrix = np.vstack(all_embeddings)
        self.add_to_faiss(embeddings_matrix, all_chunks)
        
        print(f" FAISS index now contains {self.index.ntotal} vectors")
        
        # Save everything
        self.save()
        
        # Print summary
        self.print_summary()
    
    def save(self):
        """Save FAISS index and metadata"""
        
        # Save FAISS index
        index_file = self.output_dir / "faiss.index"
        faiss.write_index(self.index, str(index_file))
        print(f"\n FAISS index saved: {index_file}")
        
        # Save metadata
        metadata_file = self.output_dir / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'chunk_id_to_idx': self.chunk_id_to_idx,
                'embedding_dim': self.embedding_dim,
                'index_type': self.faiss_index_type,
                'model': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
            }, f)
        print(f" Metadata saved: {metadata_file}")
        
        # Save stats
        stats_file = self.output_dir / "stats.json"
        stats_output = {
            **self.stats,
            'timestamp': datetime.now().isoformat(),
            'embedding_dim': self.embedding_dim,
            'index_type': self.faiss_index_type,
            'model': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
        }
        with open(stats_file, 'w') as f:
            json.dump(stats_output, f, indent=2)
        print(f" Statistics saved: {stats_file}")
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "="*70)
        print("BIOMEDBERT VECTOR DATABASE SUMMARY")
        print("="*70)
        print(f"Model: Microsoft BiomedBERT")
        print(f"\nStatistics:")
        print(f"  Files processed    : {self.stats['total_files']}")
        print(f"  Total chunks       : {self.stats['total_chunks']}")
        print(f"  Total entities     : {self.stats['total_entities']}")
        print(f"  Total vectors      : {self.index.ntotal}")
        print(f"  Embedding dim      : {self.embedding_dim}")
        print(f"  Index type         : {self.faiss_index_type.upper()}")
        
        if self.stats['entity_types']:
            print(f"\nTop Entity Types:")
            sorted_entities = sorted(self.stats['entity_types'].items(), 
                                    key=lambda x: x[1], reverse=True)
            for etype, count in sorted_entities[:15]:
                print(f"  {etype:20s} : {count:6,}")
        
        print(f"\nOutput Files:")
        print(f"  • {self.output_dir}/faiss.index")
        print(f"  • {self.output_dir}/metadata.pkl")
        print(f"  • {self.output_dir}/stats.json")
        print("="*70)


def main():
    """Main function"""
    print("= " * 25)
    print("BIOMEDBERT EMBEDDINGS → FAISS VECTOR DATABASE")
    print("= " * 25)
    
    # CONFIGURE THESE PATHS
    biomedbert_output = "/usr/users/3d_dimension_est/selva_sur/RAG/output/biomedbert_output"
    original_texts = "/usr/users/3d_dimension_est/selva_sur/RAG/output/gdc_ocr_batch/extracted_text"
    vector_db_output = "/usr/users/3d_dimension_est/selva_sur/RAG/output/biomedbert_vector_db"
    
    print(f"\nConfiguration:")
    print(f"  BiomedBERT output : {biomedbert_output}")
    print(f"  Original texts    : {original_texts}")
    print(f"  Vector DB output  : {vector_db_output}")
    print(f"\nIndex type: HNSW (balanced speed/accuracy)")
    print("="*70)
    
    try:
        pipeline = BiomedBERTToVectorDB(
            faiss_index_type="hnsw",
            output_dir=vector_db_output
        )
        
        pipeline.process_directory(
            biomedbert_output_dir=biomedbert_output,
            original_text_dir=original_texts  # Optional: for better chunking
        )
        
        print(f"\n COMPLETE: Vector database created")
        print(f" Ready for RAG queries!")
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
