import streamlit as st
from lxml import etree
from io import BytesIO
import docx
import os
from pydub import AudioSegment
import speech_recognition as sr
from utils import (
    authenticate_user, clean_text, handle_conversation, search_in_doc,
    search_web, save_text_response, speak, search_excel, search_pdf,
    get_base64_image, AudioProcessor
)

# 🔉 Voice File Processor
def process_uploaded_voice(voice_file):
    """Convert uploaded voice (.m4a/.wav) to text using SpeechRecognition."""
    import tempfile
    tmp_path = None
    wav_path = None
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
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if wav_path and wav_path != tmp_path and os.path.exists(wav_path):
            os.remove(wav_path)

# 🔧 Utility: Strip Namespace (for XML)
def strip_namespace(tag):
    return tag.split('}', 1)[1] if '}' in tag else tag

# 🧾 XML Search with Full Context
def search_large_xml(xml_content, source_tag, source_value, target_path=None):
    """
    Parse xml_content (bytes), find element(s) with tag == source_tag and text == source_value,
    then move up to the root of that subtree, and optionally extract target_path elements.
    """
    # Use an XMLParser with recovery enabled so that some malformed XML may still parse
    parser = etree.XMLParser(remove_blank_text=True, ns_clean=True, recover=True)
    tree = etree.parse(BytesIO(xml_content), parser)
    root = tree.getroot()
    results = []

    for elem in root.iter():
        # Compare stripped tag name
        if strip_namespace(elem.tag) == source_tag and elem.text and elem.text.strip() == source_value.strip():
            # Move up to top‐level context (the root of this branch)
            parent = elem
            while parent.getparent() is not None:
                parent = parent.getparent()

            if target_path:
                for target_elem in parent.iter():
                    if strip_namespace(target_elem.tag) == target_path:
                        try:
                            results.append(etree.tostring(target_elem, pretty_print=True, encoding='unicode'))
                        except Exception:
                            # fallback to text
                            results.append(target_elem.text or "")
            else:
                # return full subtree
                try:
                    results.append(etree.tostring(parent, pretty_print=True, encoding='unicode'))
                except Exception:
                    results.append(etree.tostring(parent, encoding='unicode'))

    # Remove duplicates
    unique_results = list(dict.fromkeys(results))
    return unique_results

# 🎛️ Streamlit Layout
st.set_page_config(layout="wide")

st.sidebar.title("Voice-Driven Intelligent Document Assistant")
st.sidebar.image("A-ZBlueProject/AIChatbot.png", use_container_width=True)
st.sidebar.title("🔑 User Authentication")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state["logged_in_user"] = ""

# 🔐 Authentication
if not st.session_state.authenticated:
    username_input = st.sidebar.text_input("Username:")
    password_input = st.sidebar.text_input("Password:", type="password")
    if st.sidebar.button("Login"):
        if authenticate_user(username_input, password_input):
            st.session_state.authenticated = True
            st.session_state["logged_in_user"] = username_input
            st.sidebar.success("✅ Login successful!")
            st.experimental_rerun()
        else:
            st.sidebar.error("❌ Invalid username or password.")
else:
    # Welcome message
    img_base64 = get_base64_image("A-ZBlueProject/sunita.png")
    st.markdown(f"""
        <div style="position:fixed;top:50px;right:10px;background:#333;padding:10px;border-radius:10px;color:white;">
        <img src="data:image/png;base64,{img_base64}" width="100">
        <p><b>Welcome, {st.session_state['logged_in_user']}!</b></p>
        <p style='font-size:12px;color:#ff9800;'>Created by Sunita Panicker</p></div>
    """, unsafe_allow_html=True)

    st.title("🤖 AI Doc Chatbot")

    # 🔍 Excel & PDF Search
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

    # 📄 Document & Voice Upload
    uploaded_file = st.file_uploader("Upload a Word Document (.docx)")
    voice_file = st.file_uploader("Upload a voice file (.m4a/.wav)")
    user_input = st.text_input("Ask something:")

    response = None
    doc_text = ""

    if uploaded_file:
        doc = docx.Document(uploaded_file)
        doc_text = "\n".join(p.text for p in doc.paragraphs if p.text)

    if voice_file:
        st.write("Processing voice…")
        user_input = process_uploaded_voice(voice_file)
        st.write(f"**You said:** {user_input}")

    if user_input:
        response = handle_conversation(user_input)
        if uploaded_file and doc_text:
            doc_match = search_in_doc(doc_text, user_input)
            if doc_match:
                response = doc_match
        if not response:
            search_results = search_web(user_input)
            response = "\n\n".join(search_results) if search_results else "No relevant info found."

        st.markdown(f"<div style='background:#f2f2f2;padding:10px;border-left:5px solid green;'><b>🤖 Response:</b><br>{response}</div>",
                    unsafe_allow_html=True)

    st.video("A-ZBlueProject/fixed_talking_lady.mp4")

    # 📂 DAT File Search
    st.subheader("📂 Search DAT File")
    dat_option = st.checkbox("Enable DAT Search")
    if dat_option:
        dat_file = st.file_uploader("Upload a DAT file", type=["dat"])
        search_segment = st.text_input("Enter known segment (e.g. NM1*87*2)")
        target_segment_type = st.text_input("Enter target segment (e.g. N3)")
        if st.button("Search DAT"):
            if dat_file and search_segment and target_segment_type:
                dat_content = dat_file.read().decode("utf-8", errors="ignore")
                transactions = []
                current_txn = []
                inside_txn = False
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

    # 🧾 XML Search Section
    st.subheader("🔍 XML Search with Full Context")
    xml_file = st.file_uploader("📂 Upload XML File", type=["xml"])
    if xml_file:
        st.success("✅ XML file uploaded successfully!")
        xml_content = xml_file.getvalue()
        source_tag = st.text_input("Enter source tag name (e.g., PolicyNumber):", key="xml_source_tag")
        source_value = st.text_input("Enter source tag value (e.g., H123456789):", key="xml_source_value")
        target_path = st.text_input("Enter target tag/path (optional, e.g., ClaimID, StartDate):", key="xml_target_path")
        if st.button("Search XML"):
            if source_tag and source_value:
                try:
                    results = search_large_xml(xml_content, source_tag, source_value, target_path)
                    if results:
                        st.success(f"✅ Found {len(results)} match(es):")
                        for idx, res in enumerate(results, start=1):
                            st.markdown(f"**Result {idx}:**")
                            st.code(res, language="xml")
                    else:
                        st.warning("⚠️ No matching data found.")
                except etree.XMLSyntaxError as xe:
                    st.error(f"❌ XML Syntax Error: {xe}")
                except Exception as e:
                    st.error(f"❌ Error during XML search: {e}")
            else:
                st.error("Please fill both Source Tag and Source Value before searching.")
    else:
        st.info("📄 Please upload an XML file to start searching.")
