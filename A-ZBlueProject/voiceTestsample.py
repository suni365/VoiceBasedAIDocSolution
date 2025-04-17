import streamlit as st
import streamlit.components.v1 as components
from gtts import gTTS
import base64
import os

st.set_page_config(page_title="Voice Test", layout="centered")
st.title("🎙️ Live Voice Input + Response")

# Web Speech API button
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

# Capture speech result via query
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

# ✅ NEW query param access method
voice_text = st.query_params.get('q', [''])[0]

if voice_text:
    st.markdown(f"✅ You said: **{voice_text}**")

    # Convert response to speech
    tts = gTTS(f"You said: {voice_text}")
    tts.save("output.mp3")

    with open("output.mp3", "rb") as f:
        audio_data = f.read()
        b64_audio = base64.b64encode(audio_data).decode()

    st.markdown(f"""
        <audio autoplay controls>
            <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
        </audio>
    """, unsafe_allow_html=True)

    os.remove("output.mp3")
