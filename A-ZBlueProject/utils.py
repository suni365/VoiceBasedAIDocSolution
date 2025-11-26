# utils.py
import os
import re
import base64
import numpy as np
from io import BytesIO

# --------------------------
# Authentication
# --------------------------
def authenticate_user(username, password):
    """Simple authentication (replace with your real logic)."""
    # Example: hardcoded user
    valid_users = {"sunita": "password123"}
    return valid_users.get(username) == password

# --------------------------
# Text cleaning
# --------------------------
def clean_text(text):
    """Lowercase, strip, remove unwanted chars."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# --------------------------
# Keyword-based document search
# --------------------------
def search_in_doc(doc_text, keyword):
    """Return sentences containing keyword."""
    keyword_lower = keyword.lower()
    sentences = re.split(r'(?<=[.!?]) +', doc_text)
    results = [s for s in sentences if keyword_lower in s.lower()]
    if results:
        return "\n".join(results)
    return None

# --------------------------
# Web search fallback
# --------------------------
def search_web(query):
    """Fallback web search (mock or simple)"""
    # In production, use DuckDuckGo API / Bing / Google Custom Search
    # For now, return example results
    return [f"Result 1 for '{query}'", f"Result 2 for '{query}'", f"Result 3 for '{query}'"]

# --------------------------
# Save text responses (optional)
# --------------------------
def save_text_response(filename, text):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)

# --------------------------
# Speak / Audio (GTTS optional)
# --------------------------
def speak(text, filename="response.mp3"):
    try:
        from gtts import gTTS
        tts = gTTS(text)
        tts.save(filename)
        return filename
    except Exception:
        return None

# --------------------------
# Excel search
# --------------------------
def search_excel(excel_file, keyword):
    try:
        import pandas as pd
        xls = pd.ExcelFile(excel_file)
        all_matches = []
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            df_str = df.astype(str)
            mask = df_str.apply(lambda row: row.str.contains(keyword, case=False, na=False)).any(axis=1)
            matches = df[mask]
            if not matches.empty:
                all_matches.append(matches)
        if all_matches:
            return pd.concat(all_matches)
        return pd.DataFrame()
    except Exception as e:
        return str(e)

# --------------------------
# PDF search
# --------------------------
def search_pdf(pdf_file, keyword):
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(pdf_file)
        results = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and keyword.lower() in text.lower():
                # get first matching line
                lines = text.splitlines()
                for line in lines:
                    if keyword.lower() in line.lower():
                        results.append((i+1, line.strip()))
        return results
    except Exception as e:
        return []

# --------------------------
# Base64 image helper
# --------------------------
def get_base64_image(img_path):
    if not os.path.exists(img_path):
        return ""
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return encoded

# --------------------------
# AudioProcessor placeholder
# --------------------------
class AudioProcessor:
    """Placeholder for voice processing if needed."""
    @staticmethod
    def process(file):
        return "Processed text from audio"

# --------------------------
# RAG helpers (OpenAI)
# --------------------------
def embed_texts_openai(texts, model="text-embedding-3-small"):
    """Return numpy embeddings using OpenAI."""
    try:
        import openai
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        openai.api_key = OPENAI_API_KEY
        embs = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = openai.Embeddings.create(model=model, input=batch)
            for d in resp["data"]:
                embs.append(np.array(d["embedding"], dtype=np.float32))
        return np.vstack(embs)
    except Exception as e:
        raise RuntimeError(f"OpenAI embedding failed: {e}")

def normalize(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms

def cosine_search(query_emb, corpus_emb, top_k=4):
    sims = np.dot(corpus_emb, query_emb.T).squeeze()
    top_idx = np.argsort(-sims)[:top_k]
    top_scores = sims[top_idx]
    return top_idx, top_scores

# --------------------------
# Audio / Voice file processor
# --------------------------
def process_uploaded_voice(voice_file):
    try:
        from pydub import AudioSegment
        import speech_recognition as sr
        import tempfile

        suffix = os.path.splitext(voice_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(voice_file.read())
            tmp_path = tmp_file.name

        if suffix == ".m4a":
            wav_path = tmp_path.replace(".m4a", ".wav")
            AudioSegment.from_file(tmp_path, format="m4a").export(wav_path, format="wav")
        else:
            wav_path = tmp_path

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

        return text
    except Exception as e:
        return f"Error processing voice: {e}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if wav_path != tmp_path and os.path.exists(wav_path):
            os.remove(wav_path)

# --------------------------
# XML helpers
# --------------------------
def strip_namespace(tag):
    return tag.split('}', 1)[1] if '}' in tag else tag

def search_large_xml_bytes(xml_content, source_tag, source_value, target_path=None):
    try:
        from lxml import etree
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(BytesIO(xml_content), parser)
        root = tree.getroot()
        results = []

        for elem in root.iter(source_tag):
            if elem.text and elem.text.strip() == source_value.strip():
                parent = elem
                while parent.getparent() is not None:
                    parent = parent.getparent()
                if target_path:
                    for t in parent.iter(target_path):
                        results.append(etree.tostring(t, pretty_print=True, encoding='unicode'))
                else:
                    results.append(etree.tostring(parent, pretty_print=True, encoding='unicode'))
        return results
    except Exception:
        return []
