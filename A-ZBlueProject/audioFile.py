import streamlit as st
from gtts import gTTS
import os

# Generate the audio file
text = "Hello, this is A-Z Blue Chat Bot ,how may I help you today"
tts = gTTS(text=text, lang='en')
tts.save("output.mp3")

# Streamlit UI
st.title("Download Your Audio File")

# Serve the file for download
file_path = "output.mp3"  # Adjust the path if needed
if os.path.exists(file_path):
    with open(file_path, "rb") as file:
        st.download_button(label="Download MP3", data=file, file_name="output.mp3", mime="audio/mp3")
else:
    st.error("The file 'output.mp3' was not found.")
