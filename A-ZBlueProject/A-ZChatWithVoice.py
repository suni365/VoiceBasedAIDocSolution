import streamlit as st
import docx
import os
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, WebRtcMode,AudioProcessorBase
import av
import gtts
import ffmpeg
import fitz
import base64
import time

from utils import (
    authenticate_user, clean_text, handle_conversation, search_in_doc,
    search_web, save_text_response, speak, search_excel, search_pdf,
    process_uploaded_voice, get_base64_image, AudioProcessor
)

# Set Streamlit layout
st.set_page_config(layout="wide")

# Load image and authentication UI
img_base64 = get_base64_image("A-ZBlueProject/sunita.png")
img_data_uri = f"data:image/png;base64,{img_base64}"

st.sidebar.title("Voice-Driven Intelligent Document Assistant")
st.sidebar.image("A-ZBlueProject/AIChatbot.png", use_container_width=True)
st.sidebar.title("🔑 User Authentication")

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state["logged_in_user"] = ""

# Authentication
if not st.session_state.authenticated:
    username_input = st.sidebar.text_input("Username:", key="username_input")
    password_input = st.sidebar.text_input("Password:", type="password", key="password_input")

    if st.sidebar.button("Login"):
        if authenticate_user(username_input, password_input):
            st.session_state.authenticated = True
            st.session_state["logged_in_user"] = username_input
            st.sidebar.success("✅ Login successful!")
            st.rerun()
        else:
            st.sidebar.error("❌ Invalid username or password.")

else:
    # Welcome Box
    st.markdown(
        f"""
        <style>
        .top-right {{
            position: fixed;
            top: 50px;
            right: 10px;
            background-color: #333333;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            z-index: 1000;
            text-align: center;
            width: 200px;
            color: #ffffff;
        }}
        .top-right img {{
            border-radius: 50%;
            width: 100px;
            height: 50px;
            object-fit: contain;
            object-position: top;
            margin-bottom: 10px;
        }}
        </style>
        <div class="top-right">
            <img src="data:image/png;base64,{img_base64}" alt="Profile Picture">
            <p><strong>Welcome, {st.session_state['logged_in_user']}!</strong></p>
            <p style='font-size:12px; color: #ff9800;'>Created by Sunita Panicker<br>Trivandrum, India</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Main Title
    st.title("🤖 AI Doc Chatbot")

    # Search Options
    search_option = st.sidebar.radio("Select Search Type:", ["Search Excel File", "Search PDF File"], index=0)

    if search_option == "Search Excel File":
        excel_file = st.sidebar.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
        keyword = st.sidebar.text_input("Enter keyword to search")
        if st.sidebar.button("Search Excel"):
            if excel_file and keyword:
                result = search_excel(excel_file, keyword)
                if isinstance(result, str):
                    st.sidebar.error(result)
                elif not result.empty:
                    st.sidebar.success("✅ Matches found:")
                    st.dataframe(result)
                else:
                    st.sidebar.warning("No matching data found.")
            else:
                st.sidebar.warning("Please upload a file and enter a keyword.")

    elif search_option == "Search PDF File":
        pdf_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
        keyword = st.sidebar.text_input("Enter keyword to search")
        if st.sidebar.button("Search PDF"):
            if pdf_file and keyword:
                results = search_pdf(pdf_file, keyword)
                if isinstance(results, str):
                    st.sidebar.error(results)
                elif results:
                    st.sidebar.success("✅ Matches found:")
                    for page, para in results:
                        st.sidebar.markdown(f"**📄 Page {page}:**")
                        st.sidebar.markdown(para)
                        st.sidebar.markdown("---")
                else:
                    st.sidebar.warning("No matching data found.")
            else:
                st.sidebar.warning("Please upload a file and enter a keyword.")

    # Document & Voice Uploads
    uploaded_file = st.file_uploader("Upload a document", type=["docx"])
    st.markdown("📢 **Upload a voice file:**", unsafe_allow_html=True)
    voice_file = st.file_uploader("Upload a voice file", type=["m4a", "wav"])
    user_input = st.text_input("Ask something:")

    response = None
    doc_text = ""

    # Process Document
    if uploaded_file:
        try:
            doc = docx.Document(uploaded_file)
            doc_text = "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.error(f"Error reading document: {str(e)}")

    # Process Voice File
    if voice_file:
        st.write("Processing uploaded voice...")
        user_input = process_uploaded_voice(voice_file)
        st.write(f"**You said:** {user_input}")

    # Main NLP Logic
    if user_input:
        response = handle_conversation(user_input)

        doc_match = None
        if uploaded_file:
            doc_match = search_in_doc(doc_text, user_input)

        if doc_match:
            response = doc_match
        elif not response:
            search_results = search_web(user_input)
            response = "\n\n".join(search_results) if search_results else "I'm sorry, I couldn't find relevant information."

        st.markdown(f"""
        <div style="padding:15px; border-radius:10px; background-color:#f2f2f2; color:black; border-left: 5px solid #4CAF50;">
        <b>🤖 A-Z Blue Bot Response:</b><br>{response}
        </div>""", unsafe_allow_html=True)

    # Default Video
    st.video("A-ZBlueProject/fixed_talking_lady.mp4")

    # Save files if response exists
    if "speech_file" not in st.session_state:
        st.session_state["speech_file"] = None

    if response:
        cleaned_response = clean_text(response)
        st.session_state["text_file"] = save_text_response(response)
        st.session_state["speech_file"] = speak(cleaned_response[:2000])

    # Downloads
    col1, col2, col3, col4  = st.columns(4)

    with col1:
        text_file = st.session_state.get("text_file", "")
        if text_file and os.path.exists(text_file):
            with open(text_file, "rb") as file:
                st.download_button("📄 Download Text Response", data=file, file_name="Chatbot_Response.txt", mime="text/plain")
        else:
            st.button("📄 Download Text Response", disabled=True)

    with col2:
        speech_file = st.session_state.get("speech_file", "")
        if speech_file and os.path.exists(speech_file):
            with open(speech_file, "rb") as file:
                st.download_button("🔊 Download Audio Response", data=file, file_name="Chatbot_Response.mp3", mime="audio/mp3")
        else:
            st.button("🔊 Download Audio Response", disabled=True)

    with col3:
        st.button("🎬 Download Lip-Synced Video", disabled=True)  # Placeholder
    with col4:
        ctx = None
        try:
            ctx = webrtc_streamer(
                key="live_audio",
                mode=WebRtcMode.SENDRECV,
                audio_processor_factory=AudioProcessor,
                media_stream_constraints={"audio": True, "video": False},
                async_processing=True,
                rtc_configuration={
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]}, # Free STUN
                        {
                            "urls": [
                                "turn:openrelay.metered.ca:80",
                                "turn:openrelay.metered.ca:443",
                                "turn:openrelay.metered.ca:443?transport=tcp"
                            ],
                            "username": "openrelayproject",
                            "credential": "openrelayproject"
                        }
                    ]
                }
            )
        except Exception as e:
            st.error(f"WebRTC initialization failed: {e}")

    # Correct attribute check
            if ctx and hasattr(ctx, "audio_processor") and ctx.audio_processor:
                processor = ctx.audio_processor
                if st.button("🗣️ Transcribe Live Voice"):
                    with st.spinner("Listening and transcribing..."):
                        time.sleep(3)  # Let it collect some audio
                        text = processor.get_text()
                        if text:
                            st.success(f"**You said:** {text}")
                    # Trigger the chatbot pipeline with transcribed input
                            response = handle_conversation(text)
        if uploaded_file:
            doc_match = search_in_doc(doc_text, text)
            if doc_match:
                response = doc_match

            if not response:
                search_results = search_web(text)
                response = "\n\n".join(search_results) if search_results else "Sorry, I couldn’t find anything relevant."

                # Display and speak the response
                st.markdown(f"""
                <div style="padding:15px; border-radius:10px; background-color:#f2f2f2; color:black; border-left: 5px solid #4CAF50;">
                <b>🤖 A-Z Blue Bot Response:</b><br>{response}
                </div>""", unsafe_allow_html=True)

                cleaned_response = clean_text(response)
                st.session_state["speech_file"] = speak(cleaned_response[:2000])

                with st.spinner("Generating audio response..."):
                    audio_path = st.session_state["speech_file"]
                    if audio_path and os.path.exists(audio_path):
                        audio_file = open(audio_path, "rb")
                        st.audio(audio_file.read(), format="audio/mp3")
            
        
