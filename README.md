📄 AI PDF Research Assistant
An AI-powered PDF question-answering web app that helps you quickly search, understand, and explore PDF documents using chunking, embeddings, and DeepSeek LLM.

🚀 Live App: https://aipdfresearch.streamlit.app/

✨ Features
📚 Upload any PDF document.

🧩 Automatic text chunking and tokenization.

🧠 Semantic search using FAISS and sentence-transformer embeddings.

💬 Ask natural language questions about the PDF.

⚡ Real-time streaming responses.

🔒 Secure API key handling via Streamlit Secrets.

📂 How to Use
Upload your PDF.

Wait while the app processes and embeds the content.

Type your question about the PDF in the input box.

Get an AI-generated answer along with the most relevant context chunks.

🔧 Tech Stack
Frontend: Streamlit

LLM API: DeepSeek via OpenRouter

Embeddings: sentence-transformers (all-MiniLM-L6-v2)

Vector Store: FAISS

Text Processing: LangChain Text Splitter

PDF Parsing: PyMuPDF

🛡 API Key Management
This project utilizes st.secrets to securely load API keys in Streamlit Cloud.

✍️ Author
Ateeb Zubair
Junior CS Student | Passionate about AI/ML 

🔗 GitHub Repository
https://github.com/Ateeb30/PDF-Research-AI

