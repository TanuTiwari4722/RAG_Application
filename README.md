## RAG PDF Chat Application

Chat with any PDF using this Retrieval-Augmented Generation (RAG) system built with Streamlit, FAISS, SentenceTransformers, and Hugging Face Transformers.

## Demo
Check out the live app here:  [RAG_App](https://mm98zr2dgtfj4zg5u2jg4u.streamlit.app/)

**Features**
* Semantic Search: Split PDF into chunks and search with vector embeddings.
* PDF Parsing: Extracts text from uploaded PDF files.
* Embedding Model: Uses all-MiniLM-L6-v2 from sentence-transformers.
* LLM Integration: Uses Google-flan-t5-base and Hugging Face model to generate answers.
* Interactive Chat UI: Built using Streamlit's cloud platform.
* Source Transparency: Shows exact context chunks used in answer generation.
* Fully Local (No API Key needed): Everything runs on local machine or HF Spaces.

### Tech Stack
* Component	Library/Model
* Frontend UI	Streamlit
* PDF Parsing	PyPDF2
* Embedding Generator	all-MiniLM-L6-v2 (sentence-transformers)
* Vector Store	FAISS (Inner Product Search)
* LLM for QA	google/flan-t5-base (default)
* Optional Models	DialoGPT, BART, flan-t5-large

### Setup Instructions
* Clone the repository - git clone https://github.com/yourusername/rag-pdf-chat.git
* cd rag-pdf-chat
* Create a virtual environment - python -m venv venv
* Install dependencies - pip install -r requirements.txt
* Run the application  - streamlit run app.py

### Usage Guide
* Go to the sidebar, upload your PDF.
* Click on "Process PDF" – this will extract, chunk, embed, and index the content.
* Once processed, head over to the main chat interface.
* Ask any question! Your LLM will answer using RAG technique.

### Note
* This app caches both embedding and LLM models to avoid reloading.
* First-time load may take a few minutes due to model downloads.
* GPU recommended for large models (Flan-T5-Large, BART).
