#!/usr/bin/env python3
"""
Pathology RAG System - Hugging Face Deployment
"""

import os
import sys

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Add src folder to Python path
sys.path.append('src')

from pathlib import Path

# Check if database exists
db_path = "output/biomedbert_vector_db"
if not Path(db_path).exists():
    print(f" Error: {db_path} folder not found!")
    print("Make sure output/biomedbert_vector_db/ folder is uploaded with faiss.index and metadata.pkl")
    sys.exit(1)

# Import from src folder
try:
    from retriever import CompleteRAGPipeline
    from document_processor import DynamicRAGUpdater
except ImportError as e:
    print(f" Import Error: {e}")
    print("Make sure src/retriever.py and src/dynamic_rag_updater.py exist")
    sys.exit(1)

import gradio as gr
from datetime import datetime


class PathologyRAGUI:
    """Web UI for Pathology RAG System"""
    
    def __init__(self, faiss_db_path: str, embedding_model: str):
        print("🔧 Initializing Pathology RAG System...")
        
        # Initialize RAG pipeline
        self.pipeline = CompleteRAGPipeline(
            faiss_db_path=faiss_db_path,
            embedding_model=embedding_model,
            gemini_model="gemini-pro"
        )
        
        # Initialize dynamic updater
        self.updater = DynamicRAGUpdater(
            vector_db_path=faiss_db_path,
            embedding_model=embedding_model,
            upload_dir="uploaded_reports"
        )
        
        self.faiss_db_path = faiss_db_path
        self.query_count = 0
        self.upload_count = 0
        
        print(" System ready!")
    
    def upload_pdf(self, file):
        """Handle PDF upload"""
        if file is None:
            return " No file uploaded", ""
        
        try:
            file_path = file.name
            filename = Path(file_path).name
            
            if not filename.lower().endswith('.pdf'):
                return " Only PDF files supported", ""
            
            print(f" Processing: {filename}")
            
            # Process PDF
            stats = self.updater.process_and_add_pdf(file_path)
            
            # Reload pipeline
            self.pipeline = CompleteRAGPipeline(
                faiss_db_path=self.faiss_db_path,
                embedding_model="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                gemini_model="gemini-pro"
            )
            
            self.upload_count += 1
            
            status_msg = f"""
###  Upload Successful!
**File:** {filename}  
**Time:** {stats['processing_time_seconds']:.2f}s  
**Chunks:** {stats['num_chunks']}  
**Vectors Added:** {stats['vectors_added']}  
 **Now searchable!**
            """
            
            stats_display = f"**Total Uploads:** {self.upload_count}"
            return status_msg, stats_display
            
        except Exception as e:
            return f" Error: {str(e)}", ""
    
    def process_query(self, question: str, num_sources: int = 5):
        """Process query"""
        if not question or question.strip() == "":
            return "Please enter a question.", "", ""
        
        try:
            self.query_count += 1
            result = self.pipeline.ask(question, top_k=num_sources)
            
            # Extract answer
            answer_text = result['answer']
            if "\n\nSOURCES:\n" in answer_text:
                answer = answer_text.split("\n\nSOURCES:\n")[0]
            else:
                answer = answer_text
            
            # Format sources
            sources_formatted = self._format_sources(result['sources'])
            
            # Metadata
            metadata = f"""
**Query #{self.query_count}**  
**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Sources:** {result['num_sources']}
            """
            
            return answer, sources_formatted, metadata
            
        except Exception as e:
            return f" Error: {str(e)}", "", ""
    
    def _format_sources(self, sources: list) -> str:
        """Format sources"""
        if not sources:
            return "No sources."
        
        formatted = []
        for i, source_data in enumerate(sources, 1):
            chunk = source_data['chunk']
            score = source_data.get('final_score', 0)
            
            formatted.append(f"""
### Source {i}
**File:** {chunk['filename']}  
**Score:** {score:.3f}
**Text:**
```
{chunk['text'][:300]}...
```
---
""")
        return "\n".join(formatted)
    
    def create_interface(self):
        """Create Gradio UI"""
        
        with gr.Blocks(title="Pathology RAG") as demo:
            
            gr.Markdown("""
            # 🔬 Pathology Report Analysis System
            AI-Powered Medical Document Search & Question Answering
            """)
            
            with gr.Tabs():
                
                # Upload Tab
                with gr.Tab(" Upload Report"):
                    gr.Markdown("Upload PDF files to add them to the database")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            file_upload = gr.File(label="Select PDF", file_types=[".pdf"])
                            upload_btn = gr.Button("🚀 Process & Add", variant="primary")
                        with gr.Column(scale=1):
                            upload_stats = gr.Markdown("**Stats**\n\nNo uploads yet.")
                    
                    upload_status = gr.Markdown("*Upload a PDF to begin...*")
                    
                    upload_btn.click(
                        fn=self.upload_pdf,
                        inputs=[file_upload],
                        outputs=[upload_status, upload_stats]
                    )
                
                # Query Tab
                with gr.Tab(" Search & Ask"):
                    
                    question_input = gr.Textbox(
                        label="Ask a Question",
                        placeholder="e.g., What are common breast cancer findings?",
                        lines=3
                    )
                    
                    num_sources = gr.Slider(1, 10, 5, step=1, label="Sources")
                    
                    with gr.Row():
                        submit_btn = gr.Button("🔍 Search", variant="primary")
                        clear_btn = gr.Button("Clear")
                    
                    gr.Examples(
                        examples=[
                            ["What are common findings in breast cancer pathology?"],
                            ["What are typical ER/PR/HER2 receptor patterns?"],
                            ["How is lymph node involvement assessed?"]
                        ],
                        inputs=[question_input]
                    )
                    
                    gr.Markdown("---")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            answer_output = gr.Markdown("*Your answer will appear here...*")
                        with gr.Column(scale=1):
                            metadata_output = gr.Markdown("")
                    
                    with gr.Accordion(" Sources", open=False):
                        sources_output = gr.Markdown("")
                    
                    submit_btn.click(
                        fn=self.process_query,
                        inputs=[question_input, num_sources],
                        outputs=[answer_output, sources_output, metadata_output]
                    )
                    
                    clear_btn.click(
                        fn=lambda: ("", "*Your answer will appear here...*", "", ""),
                        outputs=[question_input, answer_output, sources_output, metadata_output]
                    )
            
            gr.Markdown("---\n*For research and educational purposes only.*")
        
        return demo


def main():
    """Launch app"""
    
    print("="*70)
    print(" PATHOLOGY RAG SYSTEM")
    print("="*70)
    
    # Configuration - IMPORTANT: Use correct path!
    faiss_db_path = "output/biomedbert_vector_db"
    embedding_model = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    
    # Initialize
    ui = PathologyRAGUI(
        faiss_db_path=faiss_db_path,
        embedding_model=embedding_model
    )
    
    # Launch
    demo = ui.create_interface()
    demo.launch()


if __name__ == "__main__":
    main()
