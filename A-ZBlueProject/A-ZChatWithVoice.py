import streamlit as st
import docx
import os
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import gtts
import ffmpeg
import fitz
import base64
from utils import (
    authenticate_user, clean_text, handle_conversation, search_in_doc,
    search_web, save_text_response, speak,search_excel,search_pdf, process_uploaded_voice, get_base64_image
)

# User authentication
st.set_page_config(layout="wide")

img_base64 = get_base64_image("A-ZBlueProject/sunita.png")
img_data_uri = f"data:image/png;base64,{img_base64}"
# User authentication
st.sidebar.title("Voice-Driven Intelligent Document Assistant")
st.sidebar.image("A-ZBlueProject/AIChatbot.png", use_container_width=True)
#st.sidebar.image("AIChatbot.png",use_container_width=True)
st.sidebar.title("🔑 User Authentication")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state["logged_in_user"] = ""

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
    # Display the profile picture and welcome message in the top-right corner

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

    # Your main application code goes here
    st.title("🤖 AI Doc Chatbot")
    # 👇 THIS NEEDS TO BE INSIDE the 'else' block
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





    uploaded_file = st.file_uploader("Upload a document", type=["docx"]
                                    )
    st.markdown("📢 **Upload a voice file:**", unsafe_allow_html=True)
    voice_file = st.file_uploader("Upload a voice file", type=["m4a", "wav"])
    user_input = st.text_input("Ask something:")

    response = None
    doc_text = ""

    # Process Document File
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

    # Process User Input
    if user_input:
        response = handle_conversation(user_input)  # First check greetings

        doc_match = None
        if uploaded_file:
            doc_match = search_in_doc(doc_text, user_input)

        if doc_match:
            response = doc_match  # Prioritize document response

        # If document search fails, do web search
        elif not response:
            search_results = search_web(user_input)
            response = "\n\n".join(search_results) if search_results else None
            st.markdown(response, unsafe_allow_html=True)

        # If no response found
        if not response:
            response = "I'm sorry, I couldn't find relevant information."

        # Display chatbot response
        st.markdown(f"""
        <div style="padding:15px; border-radius:10px; background-color:#f2f2f2; color:black; border-left: 5px solid #4CAF50;">
        <b>🤖 A-Z Blue Bot Response:</b><br>{response}
        </div>""", unsafe_allow_html=True)

    # Display Default Talking Lady Video
    st.video("A-ZBlueProject/fixed_talking_lady.mp4")

    if "speech_file" not in st.session_state:
        st.session_state["speech_file"] = None

    if response:
        cleaned_response = clean_text(response)
        st.session_state["text_file"] = save_text_response(response)
        st.session_state["speech_file"] = speak(cleaned_response[:2000])

    # Always show download buttons, enable them only if the file exists
    col1, col2, col3, col4 = st.columns(4)

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
        speech_file = st.session_state.get("speech_file", "")
        if speech_file and os.path.exists(speech_file):
            st.button("Disabled for testing purpose")
            # if st.button("🎬 Download Lip-Synced Video"):
            #     # Generate lip-synced video only when button is clicked
            #     video_file = generate_lipsync_video("talking_lady.mp4", speech_file)
            #     if video_file and os.path.exists(video_file):
            #         with open(video_file, "rb") as file:
            #             st.download_button("🎬 Download Lip-Synced Video", data=file, file_name="LipSynced_Response.mp4", mime="video/mp4")
            #     else:
            #         st.error("Error generating lip-synced video.")
    with col4:
        webrtc_streamer(
        key="speech",
        mode=WebRtcMode.SENDONLY,
        client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    ),
    audio_processor_factory=AudioProcessor,
)
else:
    st.button("🎬 Download Lip-Synced Video", disabled=True)
            
