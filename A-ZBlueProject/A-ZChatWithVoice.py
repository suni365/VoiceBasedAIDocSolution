import streamlit as st
import docx
import os
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from lxml import etree
from utils import (
    authenticate_user, clean_text, handle_conversation, search_in_doc,
    search_web, save_text_response, speak, search_excel, search_pdf,
    process_uploaded_voice, get_base64_image, AudioProcessor
)

# Set Streamlit layout
st.set_page_config(layout="wide")

# Sidebar setup
st.sidebar.title("Voice-Driven Intelligent Document Assistant")
st.sidebar.image("A-ZBlueProject/AIChatbot.png", use_container_width=True)
st.sidebar.title("🔑 User Authentication")

# Initialize session
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state["logged_in_user"] = ""

# Authentication section
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

else:
    # Welcome header
    img_base64 = get_base64_image("A-ZBlueProject/sunita.png")
    st.markdown(f"""
        <div style="position:fixed;top:50px;right:10px;background:#333;padding:10px;border-radius:10px;color:white;">
        <img src="data:image/png;base64,{img_base64}" width="100">
        <p><b>Welcome, {st.session_state['logged_in_user']}!</b></p>
        <p style='font-size:12px;color:#ff9800;'>Created by Sunita Panicker</p></div>
    """, unsafe_allow_html=True)

    # Main Title
    st.title("🤖 AI Doc Chatbot")

    # ---------- Excel & PDF search ----------
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

    # ---------- Document & Voice Upload ----------
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

    # ---------- DAT Search ----------
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

   
if st.button("Search XML"):
    if xml_file and source_tag and source_value and target_path:
        with st.spinner("Searching... please wait for large XML files."):
            try:
                # Ensure the uploaded file is read in binary mode for lxml
                xml_bytes = xml_file.read()
                from io import BytesIO
                results = search_large_xml(BytesIO(xml_bytes), source_tag, source_value, target_path)

                if results:
                    st.success(f"✅ Found {len(results)} matches for '{target_path}':")
                    for res in results:
                        st.text(res)
                else:
                    st.warning("No matching data found.")

            except Exception as e:
                st.error(f"Error during XML search: {e}")
    else:
        st.error("Please fill all fields before searching.")
