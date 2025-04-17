# streamlit_app.py
import streamlit as st
import streamlit.components.v1 as components
from gtts import gTTS
import base64
import os

st.set_page_config(page_title="Voice Test", layout="centered")
st.title("🎙️ Live Voice Input + Response")

# Web Speech API button for browser-based voice input
components.html("""
    <script>
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.continuous = false;

    function startRecognition() {
        recognition.start();
        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            window.parent.postMessage({ type: 'SPEECH', text: transcript }, '*');
        };
    }
    </script>
    <button onclick="startRecognition()">🎤 Speak Now</button>
""", height=100)

# Listener for speech transcript
components.html("""
    <script>
    window.addEventListener("message", (event) => {
        if (event.data.type === 'SPEECH') {
            const msg = event.data.text;
            const newUrl = window.location.origin + window.location.pathname + '?q=' + encodeURIComponent(msg);
            window.location.href = newUrl;
        }
    });
    </script>
""", height=0)

# Get speech query from URL
voice_text = st.experimental_get_query_params().get('q', [''])[0]

# If voice input is captured
if voice_text:
    st.markdown(f"✅ You said: **{voice_text}**")

    # Generate speech using gTTS
    tts = gTTS(f"You said: {voice_text}")
    tts.save("output.mp3")

    # Convert to base64
    with open("output.mp3", "rb") as f:
        audio_data = f.read()
        b64_audio = base64.b64encode(audio_data).decode()

    # Auto-play audio response
    st.markdown(f"""
        <audio autoplay controls>
            <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
        </audio>
    """, unsafe_allow_html=True)

    os.remove("output.mp3")