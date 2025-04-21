import streamlit as st
import requests
import re
from gtts import gTTS
import os
import pandas as pd
import speech_recognition as sr
from streamlit_webrtc import AudioProcessorBase
#import imageio
#imageio.plugins.ffmpeg.download() 
import av
import os
# 
##import imageio_ffmpeg

# Set environment variable so moviepy knows where to find ffmpeg
###os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()
###from moviepy.editor import VideoFileClip, AudioFileClip, vfx
#from moviepy.video.fx import loop  # lowercase "loop"
###from moviepy.video.fx.all import loop
from pydub import AudioSegment
###import moviepy.editor as mp
#from moviepy.editor import VideoFileClip, AudioFileClip, vfx
import uuid
import fitz
import base64

# from moviepy.video.io.VideoFileClip import VideoFileClip
# from moviepy.audio.io.AudioFileClip import AudioFileClip
# from moviepy.video.fx.all import resize, fadein, fadeout

# Handle greetings

import PIL.Image

# Patch PIL.Image to support ANTIALIAS if it's missing
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

def handle_conversation(user_input):
    responses = {
        "hello": "Hello! How can I assist you today?",
        "hi": "Hello! How can I assist you today?",
        "hey": "Hello! How can I assist you today?",
        "good morning": "Hello! How can I assist you today?",
        "good evening": "Hello! How can I assist you today?",
        "how are you": "I am fine, thank you! How are you?",
        "i am good": "Good to know that! How may I assist you today?",
        "i am not good": "What happened? Is there anything I can help with?",
        "what is your name": "My name is Voice-Driven Intelligent Document Assistant",
        "where are you from": "I am from Trivandrum ,which comes under Kerala state In India ",
        "who is your favourite actor": "Aamir Khan is my favourite actor.",
        "who is your favourite actress": "My favourite actress is Madhuri Dixit.",
        "which one is your favourite color": "My favourite color is Purple.",
        "where is your favourite destination place": "My favourite destination place is Kedarnath.",
        "what is your favourite food": "My favourite food is custard.",
        "i love you": "I love you too.",
        "i like you": "I like you too.",
        "i hate you": "I still like you.",
        "you are such a darling ai": "So you too are such a lovely person.",
        "thank you so much": "You are welcome always.",
        "who created you": "I have been created by Sunita Panicker from Trivandrum.",
        "can you get me some details from the document": "For that! You need to provide  keywords from the document.",
        "Ok": "Yes ! please do go ahead with document search by using a keyword"
    }

    user_input_lower = user_input.lower()
    for key in responses:
        if key in user_input_lower:
            return responses[key]

    return None

# Authenticate user
def authenticate_user(username, password, excel_file="A-ZBlueProject/users.xlsx"):
    try:
        df = pd.read_excel(excel_file)
        df.columns = df.columns.str.strip()
        df["Username"] = df["Username"].astype(str).str.strip()
        df["Password"] = df["Password"].astype(str).str.strip()
        return ((df["Username"] == username) & (df["Password"] == password)).any()
    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")
        return False

# Search in uploaded document
def search_in_doc(doc_text, query):
    paragraphs = doc_text.split("\n")
    matching_paragraphs = [para for para in paragraphs if query.lower() in para.lower()]
    return "\n\n".join(matching_paragraphs) if matching_paragraphs else None

def search_excel(file, keyword):
    try:
        df = pd.read_excel(file, dtype=str, engine="openpyxl")
        matching_rows = df[df.apply(lambda row: row.astype(str).str.contains(keyword, case=False, na=False).any(), axis=1)]
        return matching_rows
    except Exception as e:
        return f"⚠️ Error reading the Excel file: {str(e)}"  # Removed `colored()`, replaced with plain #text


#def search_pdf(file, keyword):
    #try:
    #    extracted_data = []
    #    with fitz.open(file) as pdf:
    #        for page_num, page in enumerate(pdf, start=1):
   #             text = page.get_text("text")
   #             paragraphs = text.split("\n\n")
   #
   #         for para in paragraphs:
   #                if keyword.lower() in para.lower():
  #                      extracted_data.append((page_num, para.strip()))
  #      return extracted_data
  #  except Exception as e:
  #      return f"Error reading PDF file: {str(e)}"


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def search_pdf(pdf_file, keyword):
    try:
        pdf_doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        results = []
        for page_num in range(len(pdf_doc)):
            page = pdf_doc.load_page(page_num)
            text_dict = page.get_text("dict")
            rows = {}

            # Group spans by line Y-coordinate (approximate rows)
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        y_coord = round(line["bbox"][1], 1)
                        line_text = " ".join(span["text"] for span in line["spans"])
                        if y_coord in rows:
                            rows[y_coord] += " | " + line_text
                        else:
                            rows[y_coord] = line_text

            for y, row_text in rows.items():
                if keyword.lower() in row_text.lower():
                    results.append((page_num + 1, row_text))

        return results
    except Exception as e:
        return f"Error reading PDF file: {str(e)}"

def search_web(query):
    try:
        search_url = f"https://api.duckduckgo.com/?q={query}&format=json"
        response = requests.get(search_url)
        data = response.json()

        if "RelatedTopics" in data:
            results = [
                f"**{topic['Text']}**\n[More details]({topic['FirstURL']})"
                for topic in data["RelatedTopics"] if "Text" in topic
            ]
            return results[:3] if results else ["No relevant results found."]
        else:
            return ["No results found."]
    except Exception as e:
        return [f"Error fetching search results: {e}"]

    # Save chatbot response as text
def save_text_response(response):
    if isinstance(response, list):  # Ensure it's a list before joining
        response_text = "\n".join(response)
    else:
        response_text = response  # Keep it as-is if it's a string
    
    with open("search_results.txt", "w", encoding="utf-8") as file:
        file.write(response_text)
    
    return "search_results.txt"

def clean_text(text_list):
    cleaned_text = []

    for text in text_list:
        # Remove markdown links like [More details](URL)
        text = re.sub(r"\[.*?\]\(.*?\)", "", text)
        # Remove excessive spaces and newlines
        text = text.replace("\n", " ").strip()
        # Remove markdown bold **text**
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        cleaned_text.append(text)

    return " ".join(cleaned_text)  # Join cleaned lines into a single string

# Convert text to speech
#def speak(text):
#    speech_file = "Chatbot_Response.mp3"
#    formatted_text = text.replace("\n", " ")  # Replace newlines with spaces to form proper sentences
#    tts = gtts.gTTS(formatted_text) 
#    return speech_file 

# class AudioProcessor:
#     def __init__(self):
#         self.recognizer = sr.Recognizer()
# def recv(self, frame):
#         audio = frame.to_ndarray()
#         with sr.AudioData(audio.tobytes(), 16000, 2) as source:
#             try:
#                 text = self.recognizer.recognize_google(source)
#                 st.session_state['user_input'] = text
#             except:
#                 pass
#         return av.AudioFrame.from_ndarray(audio, layout="stereo")


class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_data = b""
        self.transcribed_text = ""

    def recv(self, frame):
        # Convert raw audio frame to bytes
        self.audio_data += frame.to_ndarray().tobytes()
        return frame

    def get_text(self):
        try:
            audio = sr.AudioData(self.audio_data, sample_rate=16000, sample_width=2)
            text = self.recognizer.recognize_google(audio)
            self.transcribed_text = text
            self.audio_data = b""  # Reset after processing
            return text
        except Exception as e:
            return f"Could not recognize speech: {str(e)}"


def speak(text):
    speech_file = "Chatbot_Response.mp3"

    if isinstance(text, list):
        text = " ".join(text)

    text = text.strip()

    # Fix spaced-out letters: "H e l l o" → "Hello"
    text = re.sub(r'(?<=\w) (?=\w)', '', text)

    print("Fixed Text for gTTS:", repr(text))  # Debugging output

    tts = gTTS(text, lang="en", slow=False)
    tts.save(speech_file)

    return speech_file
# Process uploaded voice file
# def process_uploaded_voice(audio_file):
#     try:
#         audio = AudioSegment.from_file(audio_file)  # Auto-detect format
#         wav_path = "converted_audio.wav"
#         audio.export(wav_path, format="wav")  # Convert to WAV

#         recognizer = sr.Recognizer()
#         with sr.AudioFile(wav_path) as source:
#             audio_data = recognizer.record(source)
#             return recognizer.recognize_google(audio_data)
#     except Exception as e:
#         return "Error processing voice file: " + str(e)


# def process_uploaded_voice(audio_file):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_file) as source:
#         audio = recognizer.record(source)
#     try:
#         return recognizer.recognize_google(audio)
#     except sr.UnknownValueError:
#         return "Sorry, I could not understand the audio."
#     except sr.RequestError:
#         return "Speech recognition service is unavailable."


def process_uploaded_voice(audio_file):
    recognizer = sr.Recognizer()

    # Save uploaded BytesIO to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_file_path = tmp_file.name

    # Convert .m4a to .wav using ffmpeg-python (needs ffmpeg installed)
    wav_path = tmp_file_path.replace(".m4a", ".wav")

    try:
        import ffmpeg
        ffmpeg.input(tmp_file_path).output(wav_path).run(overwrite_output=True)
    except Exception as e:
        return f"FFmpeg conversion failed: {str(e)}"

    # Now use the wav file for recognition
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError:
        return "Speech recognition service is unavailable."



#     return output_video
#def generate_lipsync_video(original_video, audio_file):
#    try:
#        # Load video and audio
#        video = VideoFileClip(original_video)
#        audio = AudioFileClip(audio_file)

        # Sync video with audio duration
#        if video.duration < audio.duration:
#            video = Loop(video, duration=audio.duration)  # Loop video if it's shorter
#        else:
#            video = video.set_duration(audio.duration)  # Trim video if it's longer

        # Set audio to video
#        video = video.set_audio(audio)

        # Save output video
#        output_video = "LipSynced_Response.mp4"
#        video.write_videofile(output_video, codec="libx264", fps=24)

#        return output_video
#    except Exception as e:
#        print(f"Error generating lip-sync video: {e}")
 #       return None  # Return None if there's an error


# def generate_lipsync_video(original_video, audio_file):
#     try:
#         # Load video and audio
#         video = mp.VideoFileClip(original_video)
#         audio = mp.AudioFileClip(audio_file).set_fps(44100)  # Ensure proper sample rate

#         # Match video duration to audio
#         if video.duration < audio.duration:
#             video = video.loop(duration=audio.duration)
#         else:
#             video = video.set_duration(audio.duration)

#         video = video.resize(height=480)

#         # Set the audio to the video
#         video = video.set_audio(audio)

#         # Generate unique output file
#         output_video = f"LipSynced_Response.mp4"

#         # Reduce CPU usage while encoding
#         video.write_videofile(
#             output_video,
#             codec="libx264",
#             fps=30,
#             audio_codec="aac",
#             preset="ultrafast",  # Faster encoding
#             threads=1,  # Reduce CPU usage 
#             verbose=True,
#             logger="bar"
#         )

#         return output_video

#     except Exception as e:
#         print(f"Error generating lip-sync video: {e}")
#         return None


