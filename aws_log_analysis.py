import streamlit as st
# import google.generativeai as genai
from PIL import Image
import PyPDF2
import docx
from google import genai



# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Debug Assistant",
    layout="wide"
)

# ---------------- API CONFIG ----------------
# genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
# # model = genai.GenerativeModel("gemini-1.5-flash")
# model = genai.GenerativeModel("gemini-1.5-flash-latest")

# ---------------- COMMON AI FUNCTION ----------------
# def analyze_with_ai(prompt, image=None):
#     try:
#         if image:
#             response = model.generate_content([image, prompt])
#         else:
#             response = model.generate_content(prompt)

#         return response.text
#     except Exception as e:
#         return f"Error: {str(e)}"


def analyze_with_ai(prompt, image=None):
    try:
        if image:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, image]
            )
        else:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

        return response.text

    except Exception as e:
        return f"Error: {str(e)}"

# ---------------- FILE READERS ----------------
def read_pdf(file):
    text = ""
    pdf = PyPDF2.PdfReader(file)
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# ---------------- SIDEBAR ----------------
st.sidebar.title("🤖 AI Assistant")

feature = st.sidebar.selectbox(
    "Choose Feature",
    [
        "🔍 Code Analyzer",
        "🐞 Error Debugger",
        "📄 Document Analyzer",
        "🖼️ Screenshot Analyzer"
    ]
)

# ---------------- CODE ANALYZER ----------------
if feature == "🔍 Code Analyzer":
    st.title("🔍 Code Analyzer")

    code = st.text_area("Paste your code here")

    if st.button("Analyze Code"):
        if code:
            with st.spinner("Analyzing code..."):
                prompt = f"""
                Analyze the following code:
                1. Errors
                2. Improvements
                3. Performance issues
                4. Security risks

                Code:
                {code}
                """
                result = analyze_with_ai(prompt)

                st.subheader("📊 Analysis Result")
                st.write(result)
        else:
            st.warning("Please enter code")

# ---------------- ERROR DEBUGGER ----------------
elif feature == "🐞 Error Debugger":
    st.title("🐞 Error Debugger")

    logs = st.text_area("Paste error logs / stack trace")

    if st.button("Debug Issue"):
        if logs:
            with st.spinner("Debugging..."):
                prompt = f"""
                You are an expert software debugger.

                Analyze this error/log:
                {logs}

                Provide:
                1. Root Cause
                2. Fix
                3. Explanation (simple)
                4. Severity
                """
                result = analyze_with_ai(prompt)

                st.subheader("🧠 Debug Result")
                st.write(result)
        else:
            st.warning("Please enter logs")

# ---------------- DOCUMENT ANALYZER ----------------
elif feature == "📄 Document Analyzer":
    st.title("📄 Document Analyzer")

    file = st.file_uploader("Upload document", type=["pdf", "docx"])

    if st.button("Analyze Document"):
        if file:
            with st.spinner("Reading document..."):
                if file.type == "application/pdf":
                    text = read_pdf(file)
                else:
                    text = read_docx(file)

            with st.spinner("Analyzing document..."):
                prompt = f"""
                Analyze this document and provide:
                - Summary
                - Key insights
                - Important points

                Document:
                {text}
                """
                result = analyze_with_ai(prompt)

                st.subheader("📘 Document Insights")
                st.write(result)
        else:
            st.warning("Please upload a file")

# ---------------- SCREENSHOT ANALYZER ----------------
elif feature == "🖼️ Screenshot Analyzer":
    st.title("🖼️ Screenshot Analyzer")

    image_file = st.file_uploader("Upload screenshot", type=["png", "jpg", "jpeg"])

    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image")

        if st.button("Analyze Screenshot"):
            with st.spinner("Analyzing screenshot..."):
                prompt = """
                Analyze this screenshot:
                - Identify error (if any)
                - Explain the issue
                - Suggest fix
                """
                result = analyze_with_ai(prompt, image=image)

                st.subheader("🧠 Screenshot Analysis")
                st.write(result)

# ---------------- FOOTER ----------------
st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ using Gemini + Streamlit")
