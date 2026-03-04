from google import genai
import streamlit as st

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])


# 🔎 Temporary Debug Test
def test_gemini():
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash-latest",
            contents="Say hello in one short sentence."
        )
        return response.text
    except Exception as e:
        return f"Gemini error: {str(e)}"


st.write("Gemini Test Result:")
st.write(test_gemini())