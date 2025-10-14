import base64
import pandas as pd
import fitz  # PyMuPDF
import re
from io import BytesIO
from PIL import Image
import openai

# 1. 🔐 User Authentication
def authenticate_user(username, password, excel_path="users.xlsx"):
    try:
        if not os.path.exists(excel_path):
            return False
        
        df = pd.read_excel(excel_path)
        
        # Normalize case and whitespace
        username = username.strip().lower()
        password = password.strip()
        
        df['username'] = df['username'].astype(str).str.strip().str.lower()
        df['password'] = df['password'].astype(str).str.strip()
        
        user_row = df[(df['username'] == username) & (df['password'] == password)]
        
        return not user_row.empty

    except Exception as e:
        print(f"Authentication error: {e}")
        return False

# 2. 🧽 Clean Text
def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

# 3. 💬 Handle Conversation
def handle_conversation(prompt):
    # Basic rule-based response
    if "hello" in prompt.lower():
        return "Hi there! How can I help you today?"
    elif "help" in prompt.lower():
        return "Sure, tell me what you need help with."
    else:
        return "I'm here to assist with document search and analysis."

# 4. 📄 Search in Word Document
def search_in_doc(doc_text, keyword):
    keyword = keyword.lower()
    matches = [line for line in doc_text.split("\n") if keyword in line.lower()]
    return "\n".join(matches) if matches else None

# 5. 🌐 Simulated Web Search (Placeholder)
def search_web(query):
    return [
        f"Result 1 for '{query}'",
        f"Result 2 for '{query}'",
        f"Result 3 for '{query}'"
    ]

# 6. 💾 Save Text Response (if needed)
def save_text_response(text, filename="response.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

# 7. 🔊 Speak Function (not implemented)
def speak(text):
    # Placeholder: could integrate TTS here
    print("Speaking:", text)

# 8. 📊 Search Excel File
def search_excel(file, keyword):
    try:
        xl = pd.ExcelFile(file)
        results = []
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            mask = df.apply(lambda row: row.astype(str).str.contains(keyword, case=False, na=False).any(), axis=1)
            filtered_df = df[mask]
            if not filtered_df.empty:
                filtered_df["Sheet"] = sheet
                results.append(filtered_df)
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
    except Exception as e:
        return f"Error reading Excel: {e}"

# 9. 📄 Search PDF File
def search_pdf(pdf_file, keyword):
    results = []
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            matches = [line.strip() for line in text.split("\n") if keyword.lower() in line.lower()]
            for match in matches:
                results.append((page_num, match))
    except Exception as e:
        return [f"Error reading PDF: {e}"]
    return results

# 10. 🖼️ Get Base64 Encoded Image (for embedding)
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        return ""

# 11. 🎙️ AudioProcessor Class (Optional)
class AudioProcessor:
    def __init__(self):
        pass

    def process(self, audio_chunk):
        # Placeholder for audio processing if needed with webrtc
        return audio_chunk

