import base64
import pandas as pd
import PyPDF2
from gtts import gTTS
import streamlit as st
import tempfile
import os

# -------------------------------
# ?? Basic Authentication
# -------------------------------
def authenticate_user(username, password):
    valid_users = {"admin": "1234", "sunita": "blue123"}
    return valid_users.get(username) == password

# -------------------------------
# ?? Text Cleaning
# -------------------------------
def clean_text(text):
    return text.strip().lower()

# -------------------------------
# ?? Simple Chat Handling
# -------------------------------
def handle_conversation(user_input):
    responses = {
        "hi": "Hello! How can I assist you today?",
        "hello": "Hi there! Ask me about your document or say something.",
        "who are you": "I’m the A-Z Blue Chat Bot, created by Sunita Panicker!"
    }
    return responses.get(clean_text(user_input), None)

# -------------------------------
# ?? Word Document Search
# -------------------------------
def search_in_doc(doc_text, keyword):
    lines = doc_text.split("\n")
    matches = [line for line in lines if keyword.lower() in line.lower()]
    return "\n".join(matches) if matches else None

# -------------------------------
# ?? Excel Search
# -------------------------------
def search_excel(excel_file, keyword):
    try:
        df = pd.read_excel(excel_file)
        mask = df.apply(lambda x: x.astype(str).str.contains(keyword, case=False, na=False))
        result = df[mask.any(axis=1)]
        return result
    except Exception as e:
        return f"Error reading Excel file: {e}"

# -------------------------------
# ?? PDF Search
# -------------------------------
def search_pdf(pdf_file, keyword):
    results = []
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if keyword.lower() in text.lower():
                results.append((i + 1, text.strip()))
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return results

# -------------------------------
# ?? Simple Web Fallback (DuckDuckGo)
# -------------------------------
def search_web(query):
    return [f"Try searching online for '{query}' (web integration placeholder)."]

# -------------------------------
# ?? Save Text to File
# -------------------------------
def save_text_response(response_text, filename="response.txt"):
    with open(filename, "w") as f:
        f.write(response_text)

# -------------------------------
# ?? Text-to-Speech
# -------------------------------
def speak(text):
    tts = gTTS(text)
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(temp_path)
    return temp_path

# -------------------------------
# ??? Image to Base64
# -------------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# -------------------------------
# ?? Audio Processor Placeholder
# -------------------------------
class AudioProcessor:
    def __init__(self):
        pass