from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import streamlit as st
import fitz, tiktoken, faiss, requests, json, os


API_KEY = st.secrets["DEEP_SEEK_API"]

# Load model with error handling
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

try:
    model = load_model()
except Exception as e:
    st.error("Failed to load the embedding model. Please check your internet connection and try again.")
    st.stop()

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text):
    tokens = encoding.encode(text)
    return len(tokens)


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_text_fast(text, max_tokens=500):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens * 4,  # Approx. 4 characters per token
        chunk_overlap=50 * 4,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks

@st.cache_data
def compute_embeddings(chunks):
    progress_bar = st.progress(0)
    chunk_embeddings = []
    batch_size = 32
    total_chunks = len(chunks)

    if total_chunks > 1000:
        st.info("This is a large PDF and might take a couple of minutes.")

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i+batch_size]
        embeddings = model.encode(batch)
        chunk_embeddings.extend(embeddings)
        progress_bar.progress(min((i + batch_size) / total_chunks, 1.0))

    st.success('Embedding complete!')
    return np.array(chunk_embeddings, dtype='float32')

@st.cache_resource
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    return faiss_index

def retrieval(user_ques, chunks, faiss_index, topk=3):
    question_embedding = model.encode([user_ques])[0]
    question_embedding = np.array([question_embedding], dtype='float32')
    distances, indices = faiss_index.search(question_embedding, topk)
    top_chunks = [chunks[i] for i in indices[0]]
    return top_chunks

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=API_KEY,
)

def ask_deepseek(user_ques, context):

    status_message = st.empty()
    status_message.info("Generating response...")  # Shows loading message


    completion = client.chat.completions.create(
    model="deepseek/deepseek-r1-0528:free",
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {user_ques}"}
        ],
        stream=True
    )
    full_response = ""
    with st.chat_message("assistant"):
        response_display = st.empty()  # Create a placeholder

        for chunk in completion:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                status_message.empty()
                response_display.markdown(full_response)  # Progressive display in Streamlit

    return full_response

# Streamlit UI
st.header("AI Research Assistant App")
st.subheader("Chunking and Tokenization")
st.sidebar.write("""
**HOW TO USE:**
1. Upload your PDF  
2. Let it process (PDFs over 500 pages take slightly longer)  
3. Ask Questions!
""")
file = st.file_uploader("Upload your PDF here", type="pdf")

if file:
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        full_text = ""
        for page_number, page in enumerate(doc, start=1):
            full_text += f"\n\n Page {page_number} \n\n"
            full_text += page.get_text()

    st.subheader("PDF Extracted successfully")
    chunks = chunk_text_fast(full_text)
    st.write(f"Total chunks = {len(chunks)}")

    with st.spinner('Embedding chunks...'):
        chunk_embeddings = compute_embeddings(chunks)

    faiss_index = build_faiss_index(chunk_embeddings)
    
    user_ques = st.text_input("Ask a question about the PDF:")
    if "summary" in user_ques.lower() or "summarize" in user_ques.lower():
        st.error("Please avoid asking for summaries. Ask specific, detailed questions instead.")
        st.stop()

    if user_ques:
        related_chunks = retrieval(user_ques, chunks, faiss_index)
        context = "\n\n".join(related_chunks)

            # NEW: Check if the retrieved chunks are too short (likely empty or not useful)
        if all(len(chunk.strip()) < 30 for chunk in related_chunks):
            st.error("The retrieved content seems insufficient. Please ask a more specific question.")

        elif num_tokens_from_string(context) > 3500:  # Adjust limit based on model's context window
            st.error("Please try asking a more specific question.")
        else:
            with st.spinner("Generating a response..."):
                llm_answer = ask_deepseek(user_ques, context)
                st.write(llm_answer)
            st.subheader("Context that answered your Question")
            for i, chunk in enumerate(related_chunks, start=1):
                st.write(f"Chunk {i}")
                st.write(chunk)
