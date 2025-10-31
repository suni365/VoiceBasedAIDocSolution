import streamlit as st
from lxml import etree
from io import BytesIO
import docx
import os
import time
from pydub import AudioSegment
import speech_recognition as sr
from io import BytesIO
import xml.etree.ElementTree as ET
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from utils import (
    authenticate_user, clean_text, handle_conversation, search_in_doc,
    search_web, save_text_response, speak, search_excel, search_pdf,
    get_base64_image, AudioProcessor
)

# --------------------------
# 🔉 Voice File Processor
# --------------------------
def process_uploaded_voice(voice_file):
    """Convert uploaded voice (.m4a/.wav) to text using SpeechRecognition."""
    import tempfile

    try:
        suffix = os.path.splitext(voice_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(voice_file.read())
            tmp_path = tmp_file.name

        if suffix == ".m4a":
            wav_path = tmp_path.replace(".m4a", ".wav")
            AudioSegment.from_file(tmp_path, format="m4a").export(wav_path, format="wav")
        else:
            wav_path = tmp_path

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

        return text

    except Exception as e:
        return f"Error processing voice: {e}"

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if wav_path != tmp_path and os.path.exists(wav_path):
            os.remove(wav_path)

# --------------------------
# 🔧 Utility: Strip Namespace (for XML)
# --------------------------
def strip_namespace(tag):
    return tag.split('}', 1)[1] if '}' in tag else tag

# --------------------------
# 🔍 Large XML Search
# --------------------------
def search_large_xml(xml_file, source_tag, source_value, target_path):
    results = []
    context = etree.iterparse(xml_file, events=("end",), recover=True)

    for event, elem in context:
        tag_name = strip_namespace(elem.tag)

        if tag_name == source_tag and (elem.text or "").strip() == source_value:
            policy_elem = elem
            while policy_elem is not None and strip_namespace(policy_elem.tag) != "PolicyInfo":
                policy_elem = policy_elem.getparent()

            if policy_elem is not None:
                if "/" in target_path:
                    try:
                        targets = policy_elem.xpath(f".//{target_path}", namespaces=None)
                        for t in targets:
                            if t.text and t.text.strip():
                                results.append(t.text.strip())
                    except Exception as e:
                        st.error(f"XPath error: {e}")
                else:
                    for t in policy_elem.iter():
                        t_name = strip_namespace(t.tag)
                        if t_name == target_path and t.text and t.text.strip():
                            results.append(t.text.strip())

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    return list(set(results))

# --------------------------
# 🎛️ Streamlit Layout
# --------------------------
st.set_page_config(layout="wide")

st.sidebar.title("Voice-Driven Intelligent Document Assistant")
st.sidebar.image("A-ZBlueProject/AIChatbot.png", use_container_width=True)
st.sidebar.title("🔑 User Authentication")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state["logged_in_user"] = ""

# --------------------------
# 🔐 Authentication
# --------------------------
if not st.session_state.authenticated:
    username_input = st.sidebar.text_input("Username:")
    password_input = st.sidebar.text_input("Password:", type="password")
    if st.sidebar.button("Login"):
        if authenticate_user(username_input, password_input):
            st.session_state.authenticated = True
            st.session_state["logged_in_user"] = username_input
            st.sidebar.success("✅ Login successful!")
            st.rerun()
        else:
            st.sidebar.error("❌ Invalid username or password.")

# --------------------------
# ✅ Main App
# --------------------------
else:
    # Welcome
    img_base64 = get_base64_image("A-ZBlueProject/sunita.png")
    st.markdown(f"""
        <div style="position:fixed;top:50px;right:10px;background:#333;padding:10px;border-radius:10px;color:white;">
        <img src="data:image/png;base64,{img_base64}" width="100">
        <p><b>Welcome, {st.session_state['logged_in_user']}!</b></p>
        <p style='font-size:12px;color:#ff9800;'>Created by Sunita Panicker</p></div>
    """, unsafe_allow_html=True)

    st.title("🤖 AI Doc Chatbot")

    # --------------------------
    # 🔍 Excel & PDF Search
    # --------------------------
    search_option = st.sidebar.radio("Select Search Type:", ["Search Excel File", "Search PDF File"])
    if search_option == "Search Excel File":
        excel_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx", "xls"])
        keyword = st.sidebar.text_input("Enter keyword")
        if st.sidebar.button("Search Excel"):
            if excel_file and keyword:
                result = search_excel(excel_file, keyword)
                if isinstance(result, str):
                    st.sidebar.error(result)
                elif not result.empty:
                    st.dataframe(result)
                else:
                    st.sidebar.warning("No matching data found.")
    else:
        pdf_file = st.sidebar.file_uploader("Upload PDF file", type=["pdf"])
        keyword = st.sidebar.text_input("Enter keyword")
        if st.sidebar.button("Search PDF"):
            if pdf_file and keyword:
                results = search_pdf(pdf_file, keyword)
                if results:
                    for page, para in results:
                        st.sidebar.markdown(f"📄 **Page {page}:** {para}")
                else:
                    st.sidebar.warning("No matching data found.")

    # --------------------------
    # 📄 Document & Voice Upload
    # --------------------------
    uploaded_file = st.file_uploader("Upload a Word Document (.docx)")
    voice_file = st.file_uploader("Upload a voice file (.m4a/.wav)")
    user_input = st.text_input("Ask something:")

    response = None
    doc_text = ""

    if uploaded_file:
        doc = docx.Document(uploaded_file)
        doc_text = "\n".join(p.text for p in doc.paragraphs)

    if voice_file:
        st.write("Processing voice...")
        user_input = process_uploaded_voice(voice_file)
        st.write(f"**You said:** {user_input}")

    if user_input:
        response = handle_conversation(user_input)
        if uploaded_file:
            doc_match = search_in_doc(doc_text, user_input)
            if doc_match:
                response = doc_match
        if not response:
            search_results = search_web(user_input)
            response = "\n\n".join(search_results) if search_results else "No relevant info found."

        st.markdown(f"<div style='background:#f2f2f2;padding:10px;border-left:5px solid green;'><b>🤖 Response:</b><br>{response}</div>", unsafe_allow_html=True)

    st.video("A-ZBlueProject/fixed_talking_lady.mp4")

    # --------------------------
    # 📂 DAT File Search
    # --------------------------
    st.subheader("📂 Search DAT File")
    dat_option = st.checkbox("Enable DAT Search")
    if dat_option:
        dat_file = st.file_uploader("Upload a DAT file", type=["dat"])
        search_segment = st.text_input("Enter known segment (e.g. NM1*87*2)")
        target_segment_type = st.text_input("Enter target segment (e.g. N3)")

        if st.button("Search DAT"):
            if dat_file and search_segment and target_segment_type:
                dat_content = dat_file.read().decode("utf-8")
                transactions, current_txn, inside_txn = [], [], False
                for line in dat_content.split("\n"):
                    for seg in line.split("~"):
                        if seg.startswith("ST*"):
                            inside_txn = True
                            current_txn = [seg]
                        elif seg.startswith("SE*"):
                            current_txn.append(seg)
                            transactions.append(current_txn)
                            inside_txn = False
                        elif inside_txn:
                            current_txn.append(seg)

                results = []
                for txn in transactions:
                    if any(seg.startswith(search_segment) for seg in txn):
                        results.extend([seg for seg in txn if seg.startswith(target_segment_type + "*")])

                if results:
                    st.success(f"✅ Found {len(results)} '{target_segment_type}' segments:")
                    for seg in results:
                        st.text(seg)
                else:
                    st.warning("No matches found.")

# --------------------------
# 🧾 Function to Search XML
# --------------------------
def search_large_xml(xml_file, source_tag, source_value, target_path=None):
    tree = etree.parse(xml_file)
    root = tree.getroot()
    results = []

    # Search for matching source tag and value
    for elem in root.iter(source_tag):
        if elem.text and elem.text.strip() == source_value.strip():
            # Find the top-level context (up to root)
            parent = elem
            while parent.getparent() is not None:
                parent = parent.getparent()

            # If target_path is specified, search for it under the same root context
            if target_path:
                for target_elem in parent.iter(target_path):
                    results.append(etree.tostring(target_elem, pretty_print=True, encoding='unicode'))
            else:
                # Return the full XML section (entire tree for that match)
                results.append(etree.tostring(parent, pretty_print=True, encoding='unicode'))

    return results

# --------------------------
# 🧾 Streamlit UI Section
# --------------------------
st.subheader("🔍 XML Search with Full Context")

# Upload XML file
xml_file = st.file_uploader("📂 Upload XML File", type=["xml"])

if xml_file:
    st.success("✅ XML file uploaded successfully!")

    # Input fields
    source_tag = st.text_input("Enter source tag name (e.g., PolicyNumber):")
    source_value = st.text_input("Enter source tag value (e.g., H123456789):")
    target_path = st.text_input("Enter target tag/path (optional, e.g., ClaimID, StartDate):")

    if st.button("Search XML"):
        if source_tag and source_value:
            try:
                xml_bytes = xml_file.read()
                results = search_large_xml(BytesIO(xml_bytes), source_tag, source_value, target_path)

                if results:
                    st.success(f"✅ Found {len(results)} match(es):")
                    for idx, res in enumerate(results, start=1):
                        st.markdown(f"**Result {idx}:**")
                        st.code(res, language="xml")
                else:
                    st.warning("⚠️ No matching data found.")
            except Exception as e:
                st.error(f"❌ Error during XML search: {e}")
        else:
            st.error("Please fill both Source Tag and Source Value before searching.")
else:
    st.info("📄 Please upload an XML file to start searching.")
