#!/usr/bin/env python3
"""
Pathology RAG System - Web UI with Dynamic Upload
Interactive interface for querying pathology reports + upload new PDFs
"""

import gradio as gr
import json
from datetime import datetime
from pathlib import Path
import shutil

# Import RAG pipeline and updater
from complete_rag_gemini import CompleteRAGPipeline
from dynamic_rag_updater import DynamicRAGUpdater


class PathologyRAGUI:
    """Web UI for Pathology RAG System with Dynamic Upload"""
    
    def __init__(self, 
                 faiss_db_path: str,
                 embedding_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"):
        """Initialize UI with RAG pipeline and dynamic updater"""
        
        print(" Initializing Pathology RAG System with Dynamic Upload...")
        
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
        
        print(" System ready!")
        
        # Statistics
        self.query_count = 0
        self.query_history = []
        self.upload_count = 0
        self.uploaded_files = []
    
    def upload_pdf(self, file) -> tuple:
        """
        Handle PDF upload, process, and update database
        
        Args:
            file: Uploaded file object from Gradio
        
        Returns:
            (status_message, stats_display) tuple
        """
        if file is None:
            return " No file uploaded", ""
        
        try:
            # Get file path
            file_path = file.name
            filename = Path(file_path).name
            
            # Check if PDF
            if not filename.lower().endswith('.pdf'):
                return f" Error: Only PDF files are supported (got {filename})", ""
            
            print(f"\n Processing uploaded file: {filename}")
            
            # Process PDF and update database
            stats = self.updater.process_and_add_pdf(file_path)
            
            # Reload pipeline to use updated database
            print(" Reloading RAG pipeline with updated database...")
            self.pipeline = CompleteRAGPipeline(
                faiss_db_path=self.faiss_db_path,
                embedding_model="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                gemini_model="gemini-pro"
            )
            print(" Pipeline reloaded")
            
            # Update stats
            self.upload_count += 1
            self.uploaded_files.append({
                'filename': filename,
                'timestamp': datetime.now().isoformat(),
                'stats': stats
            })
            
            # Format success message
            status_msg = f"""
###  Upload Successful!

**File:** {filename}  
**Processing Time:** {stats['processing_time_seconds']:.2f} seconds  
**Text Extracted:** {stats['text_length']:,} characters  
**Chunks Created:** {stats['num_chunks']}  
**Vectors Added:** {stats['vectors_added']}  

**Database Status:**
- Total Reports: {stats['total_vectors']:,} vectors
- Upload #{self.upload_count}

 **The new report is now searchable!** You can ask questions about it.
            """
            
            stats_display = f"""
**Recent Upload Statistics**

Total Uploads: {self.upload_count}  
Last Upload: {filename}  
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            return status_msg, stats_display
            
        except Exception as e:
            error_msg = f"""
###  Upload Failed

**Error:** {str(e)}

Please check:
- File is a valid PDF
- File is not corrupted
- File size is reasonable (<50MB)
            """
            return error_msg, ""
    
    def process_query(self, question: str, num_sources: int = 5) -> tuple:
        """
        Process user query and return answer
        
        Args:
            question: User's question
            num_sources: Number of source documents to retrieve
        
        Returns:
            (answer, sources, metadata) tuple
        """
        if not question or question.strip() == "":
            return "Please enter a question.", "", ""
        
        try:
            # Update stats
            self.query_count += 1
            
            # Process query
            result = self.pipeline.ask(question, top_k=num_sources)
            
            # Extract answer (remove sources section)
            answer_text = result['answer']
            if "\n\nSOURCES:\n" in answer_text:
                answer_parts = answer_text.split("\n\nSOURCES:\n")
                answer = answer_parts[0]
                sources_text = answer_parts[1] if len(answer_parts) > 1 else ""
            else:
                answer = answer_text
                sources_text = ""
            
            # Format sources with details
            sources_formatted = self._format_sources(result['sources'])
            
            # Metadata
            metadata = f"""
**Query #{self.query_count}**  
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Sources Retrieved:** {result['num_sources']}  
**Database:** {self.updater.faiss_index.ntotal:,} vectors  
**Model:** Google Gemini Pro
            """
            
            # Save to history
            self.query_history.append({
                'question': question,
                'answer': answer,
                'timestamp': datetime.now().isoformat(),
                'num_sources': result['num_sources']
            })
            
            return answer, sources_formatted, metadata
            
        except Exception as e:
            error_msg = f" Error processing query: {str(e)}"
            return error_msg, "", ""
    
    def _format_sources(self, sources: list) -> str:
        """Format source documents for display"""
        if not sources:
            return "No sources available."
        
        formatted = []
        for i, source_data in enumerate(sources, 1):
            chunk = source_data['chunk']
            score = source_data.get('final_score', 0)
            
            # Check if it's a user-uploaded file
            source_type = " NEW UPLOAD" if chunk.get('source') == 'user_upload' else "📄 DATABASE"
            
            formatted.append(f"""
### Source {i} {source_type}
**File:** `{chunk['filename']}`  
**Chunk ID:** {chunk['chunk_id']}  
**Relevance Score:** {score:.3f}  
**Entities:** {chunk.get('entity_count', 0)}

**Text Preview:**
```
{chunk['text'][:400]}{'...' if len(chunk['text']) > 400 else ''}
```
---
""")
        
        return "\n".join(formatted)
    
    def get_example_questions(self) -> list:
        """Get example questions"""
        return [
            "What are common findings in breast cancer pathology?",
            "What are typical ER/PR/HER2 receptor patterns?",
            "How is lymph node involvement assessed?",
            "What does tumor grade indicate?",
            "What are signs of metastasis in pathology reports?",
            "How is tumor size measured?",
            "What are invasive ductal carcinoma characteristics?",
            "What do surgical margins indicate?"
        ]
    
    def create_interface(self):
        """Create Gradio interface with upload capability"""
        
        # Custom CSS
        custom_css = """
        .gradio-container {
            font-family: 'Inter', sans-serif;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .upload-section {
            background-color: #f0f9ff;
            padding: 20px;
            border-radius: 10px;
            border: 2px dashed #3b82f6;
            margin-bottom: 20px;
        }
        """
        
        with gr.Blocks(css=custom_css, title="Pathology RAG System") as demo:
            
            # Header
            gr.Markdown("""
            <div class="header">
                <h1>🔬 Pathology Report Analysis System</h1>
                <p>AI-Powered Medical Document Search & Question Answering</p>
                <p style="font-size: 14px; opacity: 0.9;">Powered by BiomedBERT + Google Gemini | Real-time Database Updates</p>
            </div>
            """)
            
            # Tabs for Upload and Query
            with gr.Tabs():
                
                # Tab 1: Upload New Reports
                with gr.Tab(" Upload New Report"):
                    gr.Markdown("""
                    <div class="upload-section">
                        <h3>Upload New Pathology Report</h3>
                        <p>Upload a PDF file to add it to the searchable database. The system will:</p>
                        <ul>
                            <li> Extract text using OCR</li>
                            <li> Generate BiomedBERT embeddings</li>
                            <li> Update the vector database</li>
                            <li> Make it immediately searchable</li>
                        </ul>
                    </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            file_upload = gr.File(
                                label=" Select PDF File",
                                file_types=[".pdf"],
                                type="filepath"
                            )
                            
                            upload_btn = gr.Button(
                                " Process & Add to Database",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            upload_stats = gr.Markdown(
                                "**Upload Statistics**\n\nNo uploads yet.",
                                label="Stats"
                            )
                    
                    upload_status = gr.Markdown(
                        "*Upload a PDF to begin...*",
                        label="Status"
                    )
                    
                    # Upload button action
                    upload_btn.click(
                        fn=self.upload_pdf,
                        inputs=[file_upload],
                        outputs=[upload_status, upload_stats]
                    )
                
                # Tab 2: Query System
                with gr.Tab(" Search & Ask Questions"):
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            
                            # Question input
                            question_input = gr.Textbox(
                                label=" Ask a Question",
                                placeholder="e.g., What are common findings in breast cancer pathology?",
                                lines=3
                            )
                            
                            # Settings
                            with gr.Row():
                                num_sources = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    value=5,
                                    step=1,
                                    label=" Number of Sources to Retrieve"
                                )
                            
                            # Buttons
                            with gr.Row():
                                submit_btn = gr.Button("🔍 Search & Answer", variant="primary", size="lg")
                                clear_btn = gr.Button("🗑️ Clear", size="lg")
                            
                            # Example questions
                            gr.Markdown("###  Example Questions")
                            example_questions = self.get_example_questions()
                            gr.Examples(
                                examples=[[q] for q in example_questions],
                                inputs=[question_input],
                                label=None
                            )
                    
                    # Results section
                    gr.Markdown("---")
                    gr.Markdown("## Results")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            # Answer
                            answer_output = gr.Markdown(
                                label=" Answer",
                                value="*Your answer will appear here...*"
                            )
                        
                        with gr.Column(scale=1):
                            # Metadata
                            metadata_output = gr.Markdown(
                                label=" Query Info",
                                value=""
                            )
                    
                    # Sources section
                    with gr.Accordion(" Source Documents", open=False):
                        sources_output = gr.Markdown(
                            label="Source Documents",
                            value=""
                        )
                    
                    # Button actions
                    submit_btn.click(
                        fn=self.process_query,
                        inputs=[question_input, num_sources],
                        outputs=[answer_output, sources_output, metadata_output]
                    )
                    
                    clear_btn.click(
                        fn=lambda: ("", "*Your answer will appear here...*", "", ""),
                        inputs=[],
                        outputs=[question_input, answer_output, sources_output, metadata_output]
                    )
            
            # Footer
            gr.Markdown("""
            ---
            <div style="text-align: center; color: #666; font-size: 12px;">
                <p>⚕️ For research and educational purposes only. Not for clinical diagnosis.</p>
                <p> Questions or issues? Contact your system administrator.</p>
            </div>
            """)
        
        return demo


def main():
    """Launch the UI"""
    
    print("="*70)
    print(" PATHOLOGY RAG SYSTEM - WEB UI WITH DYNAMIC UPLOAD")
    print("="*70)
    
    # Configuration
    faiss_db_path = "/usr/users/3d_dimension_est/selva_sur/RAG/output/biomedbert_vector_db"
    embedding_model = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    
    # Initialize UI
    ui = PathologyRAGUI(
        faiss_db_path=faiss_db_path,
        embedding_model=embedding_model
    )
    
    # Create and launch interface
    demo = ui.create_interface()
    
    print("\n" + "="*70)
    print(" Launching Web Interface with Upload Support...")
    print("="*70)
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,        # Port
        share=False,             # Set True to create public link
        show_error=True
    )


if __name__ == "__main__":
    main()
