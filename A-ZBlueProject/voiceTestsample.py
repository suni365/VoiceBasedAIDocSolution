import streamlit as st
import streamlit.components.v1 as components
from gtts import gTTS
import base64
import os

st.set_page_config(page_title="Voice Test", layout="centered")
st.title("🎙️ Live Voice Input + Response")

# 🔊 JavaScript + Button for voice input
components.html("""
    <script>
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.continuous = false;

    function startRecognition() {
        console.log("🎤 Starting speech recognition...");
        recognition.start();

        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            console.log("✅ Got transcript:", transcript);
            window.postMessage({ type: 'SPEECH', text: transcript }, '*');
        };

        recognition.onerror = function(event) {
            console.error("❌ Speech recognition error:", event.error);
        };
    }

    window.addEventListener("message", (event) => {
        if (event.data.type === 'SPEECH') {
            console.log("📨 Message received:", event.data.text);
            const msg = event.data.text;
            const newUrl = window.location.origin + window.location.pathname + '?q=' + encodeURIComponent(msg);
            window.location.href = newUrl;
        }
    });
    </script>

    <button onclick="startRecognition()">🎤 Speak Now</button>
""", height=150)

# ✅ Access speech result from query string
voice_text = st.query_params.get('q', [''])[0]

if voice_text:
    st.markdown(f"✅ You said: **{voice_text}**")

    # Convert text to speech and play it
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
