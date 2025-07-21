import streamlit as st
import PyPDF2
import io
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import pickle
import os
import re
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="RAG PDF Chat Application",
    page_icon="📚",
    layout="wide"
)

class RAGSystem:
    def __init__(self):
        self.embedding_model = None
        self.llm_pipeline = None
        self.index = None
        self.chunks = []
        self.embeddings = None
        
    @st.cache_resource
    def load_embedding_model(_self):
        """Load sentence transformer model"""
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model
        except Exception as e:
            st.error(f"Error loading embedding model: {str(e)}")
            return None
    
    @st.cache_resource
    def load_llm_model(_self):
        """Load Hugging Face LLM"""
        try:
            # Better models for Q&A tasks - choose one based on your system
            
            # Option 1: Google's Flan-T5 (Best for Q&A, lightweight)
            model_name = "google/flan-t5-base"  # 250M parameters
            
            # Option 2: For more powerful responses (if you have good hardware)
            # model_name = "google/flan-t5-large"  # 780M parameters
            
            # Option 3: Microsoft's DialoGPT (conversational)
            # model_name = "microsoft/DialoGPT-small"  # 117M parameters
            
            # Option 4: Facebook's BART (good for summarization + Q&A)
            # model_name = "facebook/bart-base"
            
            # Load tokenizer and pipeline
            if "flan-t5" in model_name:
                # Text-to-text generation for Flan-T5
                pipeline_obj = pipeline(
                    "text2text-generation",
                    model=model_name,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True,
                    device=0 if torch.cuda.is_available() else -1
                )
            else:
                # Text generation for other models
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
                pipeline_obj = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=tokenizer,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True,
                    device=0 if torch.cuda.is_available() else -1
                )
            return pipeline_obj
        except Exception as e:
            st.error(f"Error loading LLM: {str(e)}")
            return None
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        # Clean the text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_text = ' '.join(words[-overlap:]) if len(words) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks"""
        if self.embedding_model is None:
            self.embedding_model = self.load_embedding_model()
        
        if self.embedding_model is None:
            return None
            
        try:
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
            return embeddings
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            return None
    
    def create_vector_store(self, embeddings: np.ndarray):
        """Create FAISS vector store"""
        try:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            index.add(embeddings.astype('float32'))
            
            return index
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None
    
    def search_similar_chunks(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Search for similar chunks using vector similarity"""
        if self.embedding_model is None or self.index is None:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search in vector store
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < len(self.chunks):
                    results.append((self.chunks[idx], float(score)))
            
            return results
        except Exception as e:
            st.error(f"Error searching chunks: {str(e)}")
            return []
    
    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """Generate answer using LLM with context"""
        if self.llm_pipeline is None:
            self.llm_pipeline = self.load_llm_model()
        
        if self.llm_pipeline is None:
            return "Sorry, LLM model is not available."
        
        try:
            # Combine context
            context = "\n".join(context_chunks[:2])  # Use top 2 chunks to avoid token limit
            
            # Different prompts for different model types
            model_name = getattr(self.llm_pipeline.model, 'name_or_path', 'unknown')
            
            if "flan-t5" in model_name.lower():
                # For Flan-T5 (text2text-generation)
                prompt = f"Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
                
                response = self.llm_pipeline(
                    prompt,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                answer = response[0]['generated_text'].strip()
                
            else:
                # For GPT-style models (text-generation)
                prompt = f"""Based on the following context, answer the question:

Context: {context}

Question: {query}

Answer:"""
                
                response = self.llm_pipeline(
                    prompt,
                    max_length=len(prompt.split()) + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.llm_pipeline.tokenizer.eos_token_id
                )
                
                # Extract the generated answer
                generated_text = response[0]['generated_text']
                answer = generated_text[len(prompt):].strip()
            
            return answer if answer else "I couldn't find a specific answer in the provided context."
            
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return "Sorry, I encountered an error while generating the answer."

# Initialize RAG system
@st.cache_resource
def get_rag_system():
    return RAGSystem()

# Main app
def main():
    st.title("RAG PDF Chat Application")
    st.markdown("Upload a PDF and chat with its contents using AI!")
    
    # Initialize RAG system
    rag = get_rag_system()
    
    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.header("Document Processing")
        
        uploaded_file = st.file_uploader(
            "Upload a PDF file", 
            type=['pdf'],
            help="Upload a PDF document to create embeddings and chat with it"
        )
        
        if uploaded_file is not None:
            st.success(f"Uploaded: {uploaded_file.name}")
            
            if st.button("Process PDF", type="primary"):
                with st.spinner("Processing PDF... This may take a few minutes"):
                    
                    # Extract text
                    st.info("Extracting text from PDF...")
                    text = rag.extract_text_from_pdf(uploaded_file)
                    
                    if text:
                        st.success(f"Extracted {len(text)} characters")
                        
                        # Chunk text
                        st.info("Splitting text into chunks...")
                        rag.chunks = rag.chunk_text(text)
                        st.success(f"Created {len(rag.chunks)} chunks")
                        
                        # Create embeddings
                        st.info("Generating embeddings...")
                        rag.embeddings = rag.create_embeddings(rag.chunks)
                        
                        if rag.embeddings is not None:
                            st.success(f"Generated embeddings: {rag.embeddings.shape}")
                            
                            # Create vector store
                            st.info("Creating vector store...")
                            rag.index = rag.create_vector_store(rag.embeddings)
                            
                            if rag.index is not None:
                                st.success("PDF processed successfully!")
                                st.session_state['pdf_processed'] = True
                            else:
                                st.error("Failed to create vector store")
                        else:
                            st.error("Failed to generate embeddings")
                    else:
                        st.error("Failed to extract text from PDF")
        
        # Display processing status
        if 'pdf_processed' in st.session_state:
            st.success("PDF Ready for Chat!")
        
        # Model info
        st.header("Model Information")
        st.info("""
        **Embedding Model**: all-MiniLM-L6-v2 (384 dim)
        **LLM Model**: google/flan-t5-base (250M params)
        **Vector Store**: FAISS with cosine similarity
        
        **Alternative Models Available:**
        - google/flan-t5-large (better quality)
        - microsoft/DialoGPT-small (conversational)
        - facebook/bart-base (summarization focus)
        """)
    
    # Main chat interface
    if 'pdf_processed' in st.session_state and st.session_state['pdf_processed']:
        st.header("Chat with your PDF")
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("View Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(source)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your PDF..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching and generating answer..."):
                    
                    # Search for relevant chunks
                    similar_chunks = rag.search_similar_chunks(prompt, k=3)
                    
                    if similar_chunks:
                        # Extract context
                        context_chunks = [chunk for chunk, score in similar_chunks]
                        
                        # Generate answer
                        answer = rag.generate_answer(prompt, context_chunks)
                        
                        st.markdown(answer)
                        
                        # Show sources
                        with st.expander("View Sources"):
                            for i, (chunk, score) in enumerate(similar_chunks, 1):
                                st.markdown(f"**Source {i} (Similarity: {score:.3f}):**")
                                st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                        
                        # Add assistant message with sources
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": context_chunks
                        })
                    else:
                        error_msg = "Sorry, I couldn't find relevant information to answer your question."
                        st.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    else:
        # Instructions when no PDF is processed
        st.header(" ****Getting Started****")
        st.markdown("""
        ### Welcome to the RAG PDF Chat Application!
        
        **Steps to use:**
        1. Upload a PDF file using the sidebar
        2. Click "Process PDF" to create embeddings
        3. Start chatting with your document!
        
        **Features:**
        - AI-powered document understanding
        - Semantic search through your PDF
        - Source citations for transparency
        - Fast vector-based retrieval
        
        **Note:** First time loading may take a few minutes to download models.
        """)

if __name__ == "__main__":
    main()