import streamlit as st
import docx
import os
import time
import speech_recognition as sr
from lxml import etree
from io import BytesIO
from pydub import AudioSegment

# 1. SET PAGE CONFIG (MUST BE FIRST)
st.set_page_config(layout="wide", page_title="AI-Chatbot")

# Install ffmpeg
os.system("apt-get install -y ffmpeg > /dev/null 2>&1")

# --- SAFE IMPORT BLOCK ---
try:
    from utils import (
        authenticate_user, clean_text, handle_conversation, search_in_doc,
        search_web, save_text_response, search_excel, search_pdf,
        get_base64_image, AudioProcessor
    )
except SyntaxError as e:
    st.error(f"❌ Syntax Error in utils.py: {e.msg} at line {e.lineno}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error importing from utils.py: {e}")
    st.stop()

# --------------------------
# 🔉 Voice File Processor
# --------------------------
def process_uploaded_voice(voice_file):
    import tempfile
    recognizer = sr.Recognizer()
    try:
        suffix = os.path.splitext(voice_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(voice_file.read())
            tmp_path = tmp_file.name

        if suffix == ".m4a":
            wav_path = tmp_path.replace(".m4a", ".wav")
            AudioSegment.from_file(tmp_path, format="m4a").export(wav_path, format="wav")
        else:
            wav_path = tmp_path

        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
            return recognizer.recognize_google(audio)
    except Exception as e:
        return f"Error: {e}"

# --------------------------
# 🔐 Authentication Logic
# --------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.sidebar.title("🔑 Login")
    username_input = st.sidebar.text_input("Username:")
    password_input = st.sidebar.text_input("Password:", type="password")
    
    if st.sidebar.button("Login"):
        if username_input and password_input:
            if authenticate_user(username_input, password_input):
                st.session_state.authenticated = True
                st.session_state["logged_in_user"] = username_input
                st.rerun()
            else:
                st.sidebar.error("❌ Invalid credentials")
        else:
            st.sidebar.warning("Please enter credentials")
    st.stop()

# --------------------------
# ✅ Main Application (Post-Login)
# --------------------------
st.sidebar.image("A-ZBlueProject/AIChatbot.png")
st.title("🤖 Intelligent AI-Chatbot")

# Sidebar User Greeting
try:
    img_base = get_base64_image("A-ZBlueProject/suntita.png.jpg")
    st.markdown(f"""<div style="position:fixed;top:50px;right:10px;background:#333;padding:10px;border-radius:10px;color:white;z-index:999;">
        <img src="data:image/png;base64,{img_base}" width="50"><br><b>{st.session_state['logged_in_user']}</b></div>""", unsafe_allow_html=True)
except: 
    pass

# --------------------------
# 🔍 Main Document & Voice Search
# --------------------------
st.header("🔍 Document & Voice Search")

# Added unique keys here to prevent duplicate ID errors
uploaded_file = st.file_uploader("Upload Word Document (.docx)", type="docx", key="doc_search_uploader")
user_input = st.text_input("Enter keyword or phrase to search in Doc:")
voice_file = st.file_uploader("OR Upload Voice (.m4a/.wav)", key="voice_search_uploader")

response = ""

# 1. Handle Voice Transcription
if voice_file:
    with st.spinner("Transcribing..."):
        voice_text = process_uploaded_voice(voice_file)
        if voice_text and not voice_text.startswith("Error"):
            user_input = voice_text
            st.success(f"Captured: {user_input}")
        else:
            st.error(voice_text)

# 2. Search Logic (Paragraph based)
if uploaded_file and user_input:
    doc = docx.Document(uploaded_file)
    target = user_input.strip().lower()
    matches = []
    
    for para in doc.paragraphs:
        if target in para.text.lower():
            if para.text.strip():
                matches.append(para.text)
    
    if matches:
        st.subheader("Document Matches")
        for m in matches:
            st.info(m)
        response = matches[0] # Use first match for AI context
    else:
        st.warning("Phrase not found in document.")

# 3. AI / Web Fallback
if user_input and not response:
    with st.spinner("Consulting AI..."):
        response = handle_conversation(user_input)
        if not response or "No relevant info" in response:
            web_res = search_web(user_input)
            response = "\n\n".join(web_res) if web_res else "No info found."

if response:
    st.markdown(f"<div style='background:#f9f9f9;padding:15px;border-left:5px solid #007bff;color:black;'><b>🤖 AI Response:</b><br>{response}</div>", unsafe_allow_html=True)
    try:
        st.video("A-ZBlueProject/fixed_talking_lady.mp4")
    except: 
        pass

st.divider()

# --- Optional Sidebar Search (Excel/PDF) ---
st.sidebar.subheader("Additional Search")
search_option = st.sidebar.radio("File Search:", ["Excel", "PDF"])
side_key = st.sidebar.text_input("Sidebar Keyword")
if st.sidebar.button("Search Sidebar File"):
    if search_option == "Excel":
        res = search_excel(side_key)
    else:
        res = search_pdf(side_key)
    st.sidebar.write(res)