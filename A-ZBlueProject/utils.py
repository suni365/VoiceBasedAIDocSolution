import os
import re
import base64
import numpy as np
from io import BytesIO
import streamlit as st

# --------------------------
# 🔐 Authentication
# --------------------------
def authenticate_user(username, password):
    valid_users = {"sunita": "password123"}
    return valid_users.get(username) == password


# --------------------------
# 🧹 Text Cleaning
# --------------------------
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


# --------------------------
# 🤖 Chatbot Core Logic (Lightweight GPT-like Behavior)
# --------------------------
def handle_conversation(prompt):
    prompt = prompt.lower()

    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if any(greet in prompt for greet in greetings):
        return "Hello! How can I assist you today?"

    if "how are you" in prompt:
        return "I am fine, thank you and How are you ?"

    if "I am good" in prompt:
        return "Good to know that, how may I assistant you today"
        
    if "who are you" in prompt:
        return "I am your AI-powered document assistant."

    if "what can you do" in prompt:
        return "I can search documents, analyze PDFs, Excel files, XML, and respond using voice or text."

    if "thank" in prompt:
        return "You're welcome! 😊"

    return ""   # fallback → document + web search


# --------------------------
# 📄 Word Document Search
# --------------------------
def search_in_doc(doc_text, keyword):
    keyword_lower = keyword.lower()
    sentences = re.split(r'(?<=[.!?]) +', doc_text)
    results = [s for s in sentences if keyword_lower in s.lower()]
    return "\n".join(results) if results else None


# --------------------------
# 🌐 Web Search Fallback (Simple Mock)
# --------------------------
def search_web(query):
    return [
        f"Top result for '{query}'",
        f"Additional reference for '{query}'",
        f"More information about '{query}'"
    ]


# --------------------------
# 📝 Save Text Response
# --------------------------
def save_text_response(filename, text):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)


# --------------------------
# 📊 Excel Search
# --------------------------
def search_excel(excel_file, keyword):
    try:
        import pandas as pd
        xls = pd.ExcelFile(excel_file)
        all_matches = []

        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            df_str = df.astype(str)
            mask = df_str.apply(
                lambda row: row.str.contains(keyword, case=False, na=False)
            ).any(axis=1)
            matches = df[mask]
            if not matches.empty:
                all_matches.append(matches)

        return pd.concat(all_matches) if all_matches else pd.DataFrame()

    except Exception as e:
        return str(e)


# --------------------------
# 📕 PDF Search
# --------------------------
def search_pdf(pdf_file, keyword):
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(pdf_file)
        results = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and keyword.lower() in text.lower():
                for line in text.splitlines():
                    if keyword.lower() in line.lower():
                        results.append((i + 1, line.strip()))

        return results

    except Exception:
        return []


# --------------------------
# 🖼️ Base64 Image Helper
# --------------------------
def get_base64_image(img_path):
    if not os.path.exists(img_path):
        return ""
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return encoded


# --------------------------
# 🎤 Audio Processor Placeholder
# --------------------------
class AudioProcessor:
    @staticmethod
    def process(file):
        return "Audio processed successfully"


# --------------------------
# 🧾 XML Utilities
# --------------------------
def strip_namespace(tag):
    return tag.split('}', 1)[1] if '}' in tag else tag



