import streamlit as st
import docx
import PyPDF2
import pandas as pd
import xml.etree.ElementTree as ET
# import os  # Only needed if you use audio/video
# from pydub import AudioSegment
# import speech_recognition as sr

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# ==========================
# UTILITY FUNCTIONS
# ==========================

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_excel(file):
    df = pd.read_excel(file)
    return df.to_string()

def read_dat(file):
    return file.read().decode(errors="ignore")

def read_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()
    return ET.tostring(root, encoding='unicode')

# ==========================
# RAG / NLP SETUP
# ==========================

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variables for RAG
index = None
text_store = []

def build_rag_index(text_chunks):
    """Build FAISS index from text chunks"""
    global index, text_store
    text_store = text_chunks
    embeddings = model.encode(text_chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

def rag_query(question, top_k=3):
    """Query RAG index and return top_k relevant chunks"""
    if index is None or not text_store:
        return "RAG index not built yet. Upload documents first."
    
    question_emb = model.encode([question])
    D, I = index.search(question_emb, top_k)
    results = [text_store[i] for i in I[0]]
    return "\n\n".join(results)

# ==========================
# STREAMLIT UI
# ==========================

st.title("A-Z Chatbot (RAG Enabled, Audio/Video Disabled)")

uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)

all_text_chunks = []

if uploaded_files:
    for file in uploaded_files:
        st.write(f"Processing {file.name}...")
        if file.name.lower().endswith(".docx"):
            text = read_docx(file)
        elif file.name.lower().endswith(".pdf"):
            text = read_pdf(file)
        elif file.name.lower().endswith(".xlsx"):
            text = read_excel(file)
        elif file.name.lower().endswith(".dat"):
            text = read_dat(file)
        elif file.name.lower().endswith(".xml"):
            text = read_xml(file)
        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue

        # Chunk text into sentences or paragraphs (~40+ chars)
        for chunk in text.split(". "):
            if len(chunk.strip()) > 40:
                all_text_chunks.append(chunk.strip())

# Build RAG index
if all_text_chunks:
    build_rag_index(all_text_chunks)
    st.success("✅ RAG index built successfully! You can now ask natural-language questions.")

    question = st.text_input("Ask anything based on uploaded documents:")

    if question:
        response = rag_query(question)
        st.write("### Response:")
        st.write(response)
        # -------------------------
        # Audio/Video response (commented out)
        # -------------------------
        # speak(response)
        # generate_video_response(response)
