import streamlit as st
from google import genai

st.title("AI Log Analyzer")

log_text = st.text_area("Paste your log here")

uploaded_file = st.file_uploader("Or upload a log file")

if uploaded_file:
    log_text = uploaded_file.read().decode("utf-8")


def extract_errors(log_text):

    lines = log_text.split("\n")
    errors = []

    for line in lines:
        if "error" in line.lower() or "exception" in line.lower():
            errors.append(line)

    return "\n".join(errors)


client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


def analyze_log(log_text):

    prompt = f"""
    Analyze the following system log.

    Identify:
    1. Root cause of the error
    2. Which component is failing
    3. Suggested fix

    Log:
    {log_text}
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt
    )

    return response.text


if st.button("Analyze Log"):

    if log_text.strip() == "":
        st.warning("Please paste or upload logs first.")

    else:
        errors = extract_errors(log_text)

        if errors.strip() == "":
            st.info("No obvious error lines found. Sending full log for analysis.")
            errors = log_text

        try:
            result = analyze_log(errors)
            st.write(result)

        except Exception as e:
            st.error(f"AI analysis failed: {e}")