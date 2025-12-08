import os
import re
import base64
import numpy as np
from io import BytesIO

# --------------------------
# 🧠 LLM/Chatbot Interaction
# --------------------------
def handle_conversation(prompt, context=None): # <--- CHANGE 1: Add a context parameter
    """
    Interacts with the Gemini model to generate a conversational or synthetic response.
    If context is provided (from RAG), it uses the context to answer the prompt naturally.
    """
    try:
        from google import genai
        
        client = genai.Client()
        
        # 2. Define the system instruction/context for the LLM
        # <--- CHANGE 2: Tweak the instruction for RAG-specific behavior
        system_instruction = (
            "You are an expert document analysis assistant and chatbot. "
            "Your task is to answer the user's question using ONLY the provided document CONTEXT. "
            "If the context is provided, you must synthesize and rephrase the information "
            "into a smooth, natural, and direct answer. Do not just copy the source text. "
            "If the context does not contain the answer, state politely that the information is not found in the documents. "
            "If no context is provided, respond conversationally as a general chatbot."
        )

        # <--- CHANGE 3: Combine prompt and context for the LLM input
        full_prompt = prompt
        if context:
            full_prompt = (
                f"CONTEXT:\n---\n{context}\n---\n\n"
                f"USER QUESTION: {prompt}\n\n"
                f"Please generate a natural language response based on the CONTEXT that directly answers the USER QUESTION."
            )

        # 3. Generate the response
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt, # <--- Sending the combined prompt now
            config={
                'system_instruction': system_instruction
            }
        )
        
        return response.text

    except ImportError:
        return f"Error: The 'google-genai' library is required for the LLM function. Please install it."
    except Exception as e:
        return f"LLM Connection Error: Could not generate response. Details: {e}"

# --------------------------
# 🔐 Authentication
# --------------------------
def authenticate_user(username, password):
    """Simple authentication."""
    valid_users = {"sunita": "password123"}
    return valid_users.get(username) == password


# --------------------------
# 🧹 Text Cleaning
# --------------------------
def clean_text(text):
    """Lowercase, strip, remove extra spaces."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


# --------------------------
# 📄 Keyword-based Document Search
# --------------------------
def search_in_doc(doc_text, keyword):
    """Return sentences containing the keyword."""
    keyword_lower = keyword.lower()
    sentences = re.split(r'(?<=[.!?]) +', doc_text)
    results = [s for s in sentences if keyword_lower in s.lower()]
    return "\n".join(results) if results else None


# --------------------------
# 🌐 Web Search Fallback (Simple Mock)
# --------------------------
def search_web(query):
    return [
        f"Result 1 for '{query}'",
        f"Result 2 for '{query}'",
        f"Result 3 for '{query}'"
    ]


# --------------------------
# 📝 Save Text Response
# --------------------------
def save_text_response(filename, text):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)


# --------------------------
# 🔉 Text → Speech (GTTS)
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
# 🔉 AudioProcessor Placeholder
# --------------------------
class AudioProcessor:
    @staticmethod
    def process(file):
        return "Processed text from audio"


# --------------------------
# 🧠 OpenAI RAG Embeddings
# --------------------------
def embed_texts_openai(texts, model="text-embedding-3-small"):
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
# 🎤 Voice File Processor
# --------------------------
def process_uploaded_voice(voice_file):
    """
    Converts m4a/wav → text using SpeechRecognition.
    """
    try:
        from pydub import AudioSegment
        import speech_recognition as sr
        import tempfile

        suffix = os.path.splitext(voice_file.name)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(voice_file.read())
            tmp_path = tmp_file.name

        # Convert if needed
        if suffix == ".m4a":
            wav_path = tmp_path.replace(".m4a", ".wav")
            AudioSegment.from_file(tmp_path, format="m4a").export(wav_path, format="wav")
        else:
            wav_path = tmp_path

        recog = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recog.record(source)
            text = recog.recognize_google(audio)

        return text

    except Exception as e:
        return f"Error processing voice: {e}"

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if 'wav_path' in locals() and wav_path != tmp_path and os.path.exists(wav_path):
            os.remove(wav_path)


# --------------------------
# 🧾 XML Utilities
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


