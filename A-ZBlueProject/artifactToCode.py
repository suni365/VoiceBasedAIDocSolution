import streamlit as st
from google import genai
from PIL import Image

# ---------------------------
# Secure API Configuration
# ---------------------------
# Make sure you added this in Streamlit Cloud → Settings → Secrets:
# GOOGLE_API_KEY = "your_new_key"

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])


# ---------------------------
# AI Function
# ---------------------------
def analyze_code_artifact(image, error_text):

    prompt = f"""
    You are an expert Senior Developer.
    Analyze the provided screenshot of code and the error log below.

    Error Log:
    {error_text}

    Tasks:
    1. Pinpoint the exact line number or code block causing the issue.
    2. Explain the technical reason (Syntax, Logic, or Dependency).
    3. Provide the corrected code snippet.
    """
    try:
        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents="Say hello in one sentence"
        )
        st.write(response.text)
        return response.text
    except Exception as e:
        return f"Error connecting to Gemini API: {str(e)}"


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Artifact Debugger", layout="wide")

st.title("🚀 AI Code Artifact & Error Reviewer")
st.write("Upload a screenshot of your code and paste the error to get a fix.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "1. Upload Code Screenshot",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Artifact", use_container_width=True)

with col2:
    error_input = st.text_area(
        "2. Paste the Error/Log here",
        height=200
    )

    if st.button("Analyze & Fix"):

        if uploaded_file and error_input:
            with st.spinner("Analyzing artifact and logs..."):
                result = analyze_code_artifact(img, error_input)

            st.markdown("### 💡 Analysis & Solution")
            st.info(result)

        else:
            st.warning("Please provide both an image and an error log.")


