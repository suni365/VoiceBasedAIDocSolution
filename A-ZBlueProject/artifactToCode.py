import streamlit as st 
import google.generativeai as genai 
from PIL import Image 
import os 
# --- Configuration --- # 
# It is safer to use secrets or environment variables 
API_KEY = "AIzaSyBVqeS6v4aPSUS6NeQ8h78HBUk4519gbKU"  
genai.configure(api_key=API_KEY) 

# Use 1.5-flash for speed/cost, or  1.5-pro for complex logic 

model = genai.GenerativeModel('gemini-1.5-flash') 

def analyze_code_artifact(image, error_text): 
    prompt = f""" 
    You are an expert Senior Developer. 
    Analyze the provided image (code artifact) and the error log below. 

    Error Log: 
    {error_text} 

   Tasks: 
   1. Pinpoint the exact line number or code block in the image causing the issue. 
   2. Explain the technical reason for the failure (e.g., Syntax, Logic, or Dependency). 
   3. Provide the corrected code snippet. 
   """ 
   try: 
       # The API accepts a list: [prompt, image] 
       response = model.generate_content([prompt, image]) 
       return response.text 
   except Exception as e: 
       return f"Error connecting to Gemini API: {str(e)}" # --- Streamlit UI --- 
st.set_page_config(page_title="Artifact Debugger", layout="wide") 
st.title("🚀 AI Code Artifact & Error Reviewer") 
st.write("Upload a screenshot of your code and paste the error to get a fix.") 

col1, col2 = st.columns(2)

with col1: 
    uploaded_file = st.file_uploader("1. Upload Code Screenshot", type=["png", "jpg", "jpeg"]) 
    if uploaded_file: 
        img = Image.open(uploaded_file) 
        st.image(img, caption="Uploaded Artifact", use_container_width=True) 

with col2: 
    error_input = st.text_area("2. Paste the Error/Log here", height=200) 
    if st.button("Analyze &  & Fix"): 
        if uploaded_file and error_input: 
            with st.spinner("Analyzing artifact and logs..."): 
                result = analyze_code_artifact(img, error_input) 
                st.markdown("### 💡 Analysis & Solution") 
                st.info(result) 
        
        else: st.warning("Please provide both an image and an error log.")







