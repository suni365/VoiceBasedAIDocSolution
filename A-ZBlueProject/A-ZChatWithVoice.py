import streamlit as st
from lxml import etree
from io import BytesIO
import docx
import os
import time
import xml.etree.ElementTree as ET
from pydub import AudioSegment
import speech_recognition as sr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from utils import (
    authenticate_user, clean_text, handle_conversation, search_in_doc,
    search_web, save_text_response, speak, search_excel, search_pdf,
    get_base64_image, AudioProcessor )
# --------------------------
# App config
# --------------------------
st.set_page_config(layout="wide")
MODEL_NAME = "all-MiniLM-L6-v2"  # lightweight Sentence-Transformers model

# --------------------------
# Helper: Text extraction
# --------------------------
# Trigger rebuild
def extract_text_from_docx(file_bytes):
    try:
        doc = docx.Document(BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def extract_text_from_pdf(file_bytes):
    try:
        # try PyPDF2
        from PyPDF2 import PdfReader
        reader = PdfReader(BytesIO(file_bytes))
        texts = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                texts.append(txt)
        return "\n".join(t for t in texts if t)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_excel(file_bytes):
    try:
        xls = pd.read_excel(BytesIO(file_bytes), sheet_name=None, dtype=str)
        texts = []
        for sheet_name, df in xls.items():
            df = df.fillna("")
            # join rows into lines
            for _, row in df.iterrows():
                row_text = " | ".join(str(v) for v in row.values if v != "")
                if row_text.strip():
                    texts.append(row_text)
        return "\n".join(texts)
    except Exception as e:
        st.error(f"Error reading Excel: {e}")
        return ""

# --------------------------
# Helper: Chunking + Embeddings
# --------------------------
def chunk_text(text, chunk_size=400, overlap=50):
    """Simple whitespace chunking, returns list of chunks."""
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += (chunk_size - overlap)
    return chunks

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)

def embed_texts(model, texts):
    """Return numpy array of embeddings for a list of texts."""
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()))
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embs

def semantic_search(query, chunks, chunk_embeddings, model, top_k=3):
    """Return top_k chunks by cosine similarity to query embedding."""
    if not chunks or chunk_embeddings.size == 0:
        return []
    q_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    sims = cosine_similarity(q_emb, chunk_embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_idx:
        results.append({"chunk": chunks[idx], "score": float(sims[idx]), "index": int(idx)})
    return results

# --------------------------
# Audio processing
# --------------------------
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
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            if wav_path and wav_path != tmp_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

# --------------------------
# XML helpers (single definition)
# --------------------------
def strip_namespace(tag):
    return tag.split('}', 1)[1] if '}' in tag else tag

def search_large_xml_bytes(xml_bytes, source_tag, source_value, target_path=None):
    """Search XML bytes for source tag/value; return matching target path(s) or full context."""
    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    try:
        tree = etree.parse(BytesIO(xml_bytes), parser)
    except Exception as e:
        raise
    root = tree.getroot()
    results = []
    for elem in root.iter():
        if strip_namespace(elem.tag) == source_tag and elem.text and elem.text.strip() == source_value.strip():
            parent = elem
            # climb to document root of that section
            while parent.getparent() is not None:
                parent = parent.getparent()
            if target_path:
                for t in parent.iter(target_path):
                    results.append(etree.tostring(t, pretty_print=True, encoding='unicode'))
            else:
                results.append(etree.tostring(parent, pretty_print=True, encoding='unicode'))
    return results

# --------------------------
# UI: Sidebar - Auth + Search toggles
# --------------------------
st.sidebar.title("Voice-Driven Intelligent Document Assistant")
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state["logged_in_user"] = ""

st.sidebar.title("🔑 User Authentication")
if not st.session_state.authenticated:
    username_input = st.sidebar.text_input("Username:")
    password_input = st.sidebar.text_input("Password:", type="password")
    if st.sidebar.button("Login"):
        if authenticate_user(username_input, password_input):
            st.session_state.authenticated = True
            st.session_state["logged_in_user"] = username_input
            st.sidebar.success("✅ Login successful!")
            # st.experimental_rerun()
        else:
            st.sidebar.error("❌ Invalid username or password.")
else:
    st.sidebar.success(f"Logged in as {st.session_state['logged_in_user']}")

# --------------------------
# Main app layout
# --------------------------
if st.session_state.get("authenticated", False):
    img_base64 = get_base64_image("A-ZBlueProject/sunita.png") if os.path.exists("A-ZBlueProject/sunita.png") else None
    if img_base64:
        st.markdown(f"""
            <div style="position:fixed;top:50px;right:10px;background:#333;padding:10px;border-radius:10px;color:white;">
            <img src="data:image/png;base64,{img_base64}" width="100">
            <p><b>Welcome, {st.session_state['logged_in_user']}!</b></p>
            <p style='font-size:12px;color:#ff9800;'>Created by Sunita Panicker</p></div>
        """, unsafe_allow_html=True)

    st.title("🤖 AI Doc Chatbot (RAG-enabled)")

    # Sidebar: Excel / PDF quick search (kept)
    search_option = st.sidebar.radio("Select Search Type:", ["Search Excel File", "Search PDF File", "No Quick Search"])
    if search_option == "Search Excel File":
        excel_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx", "xls"])
        keyword = st.sidebar.text_input("Enter keyword", key="excel_kw")
        if st.sidebar.button("Search Excel"):
            if excel_file and keyword:
                result = search_excel(excel_file, keyword)
                if isinstance(result, str):
                    st.sidebar.error(result)
                elif not result.empty:
                    st.dataframe(result)
                else:
                    st.sidebar.warning("No matching data found.")
    elif search_option == "Search PDF File":
        pdf_file = st.sidebar.file_uploader("Upload PDF file", type=["pdf"])
        keyword = st.sidebar.text_input("Enter keyword for PDF", key="pdf_kw")
        if st.sidebar.button("Search PDF"):
            if pdf_file and keyword:
                results = search_pdf(pdf_file, keyword)
                if results:
                    for page, para in results:
                        st.sidebar.markdown(f"📄 **Page {page}:** {para}")
                else:
                    st.sidebar.warning("No matching data found.")

    # --------------------------
    # Document upload area (multiple files allowed)
    # --------------------------
    st.subheader("📄 Upload Documents for AI Search (DOCX / PDF / XLSX)")
    uploaded_files = st.file_uploader("Upload Documents (you can upload multiple)", type=["docx", "pdf", "xlsx", "xls"], accept_multiple_files=True)

    # RAG engine state: store per-file chunks and embeddings
    if "rag_store" not in st.session_state:
        st.session_state.rag_store = {}  # key: filename, value: dict with 'chunks' and 'embeddings' and 'text'

    model = load_embedding_model()

    # Process newly uploaded files and build embeddings if not already present
    if uploaded_files:
        for f in uploaded_files:
            fname = f.name
            if fname in st.session_state.rag_store:
                continue  # already processed
            content = f.read()
            # Extract depending on type
            if fname.lower().endswith(".docx"):
                text = extract_text_from_docx(content)
            elif fname.lower().endswith(".pdf"):
                text = extract_text_from_pdf(content)
            elif fname.lower().endswith((".xlsx", ".xls")):
                text = extract_text_from_excel(content)
            else:
                text = ""

            text = clean_text(text) if text else ""
            if not text:
                st.warning(f"No extractable text from {fname}. Skipping RAG for this file.")
                st.session_state.rag_store[fname] = {"text": "", "chunks": [], "embeddings": np.zeros((0, model.get_sentence_embedding_dimension()))}
                continue

            chunks = chunk_text(text, chunk_size=300, overlap=60)
            embeddings = embed_texts(model, chunks)

            st.session_state.rag_store[fname] = {"text": text, "chunks": chunks, "embeddings": embeddings}

        st.success("✅ Documents processed for RAG search.")

    # --------------------------
    # Voice / Text Query Input
    # --------------------------
    st.subheader("🔊 Ask (type or upload voice)")
    col1, col2 = st.columns([3,1])
    with col1:
        user_input = st.text_input("Ask something:", key="user_question")
    with col2:
        voice_file = st.file_uploader("Upload voice (.m4a/.wav) for question", type=["m4a", "wav"], key="voice_question")
        if st.button("Use voice to ask"):
            if voice_file:
                with st.spinner("Processing voice..."):
                    q_text = process_uploaded_voice(voice_file)
                    st.session_state.user_question = q_text
                    user_input = q_text
                    st.success(f"You said: {q_text}")
            else:
                st.warning("Please upload a voice file first.")

    # --------------------------
    # When user asks something
    # --------------------------
    if user_input and user_input.strip():
        question = user_input.strip()
        st.markdown(f"**Your question:** {question}")

        # 1) Query RAG across all uploaded files (if any)
        rag_results_combined = []
        for fname, entry in st.session_state.rag_store.items():
            if not entry.get("chunks"):
                continue
            results = semantic_search(question, entry["chunks"], entry["embeddings"], model, top_k=3)
            for r in results:
                r_copy = r.copy()
                r_copy["source_file"] = fname
                rag_results_combined.append(r_copy)

        # Sort combined results by score desc
        rag_results_combined = sorted(rag_results_combined, key=lambda x: x["score"], reverse=True)

        # 2) If RAG found something above threshold, present best matches and return concatenated answer
        ANSWER_SCORE_THRESHOLD = 0.35  # tuneable; lower if you want looser matches
        final_answer = None
        if rag_results_combined and rag_results_combined[0]["score"] >= ANSWER_SCORE_THRESHOLD:
            # Present top 3 uniquely
            shown = 0
            st.success("🔎 Found relevant excerpts from uploaded documents (RAG):")
            for hit in rag_results_combined[:5]:
                shown += 1
                st.markdown(f"**Source:** {hit['source_file']}  —  **Score:** {hit['score']:.3f}")
                st.write(hit["chunk"])
                st.markdown("---")
            # Optionally call your handle_conversation or just return the top chunk as answer
            # top_chunks_text = "\n\n".join(h["chunk"] for h in rag_results_combined[:3])
            # final_answer = top_chunks_text
            top_chunks_context = "\n\n".join(h["chunk"] for h in rag_results_combined[:3])
            with st.spinner("Synthesizing answer from documents..."):
                final_answer = handle_conversation(question, context=top_chunks_context)
        else:
            # 3) Fallback to your existing keyword-based doc search across combined document text
            st.info("No strong RAG match — falling back to keyword search + LLM response.")
            # Combine all doc text to search_in_doc (if available)
            combined_doc_text = "\n\n".join(entry["text"] for entry in st.session_state.rag_store.values() if entry.get("text"))
            doc_match = ""
            if combined_doc_text:
                try:
                    doc_match = search_in_doc(combined_doc_text, question)  # your original function
                except Exception as e:
                    st.error(f"Error in search_in_doc: {e}")
            if doc_match:
                final_answer = doc_match
            else:
                # 4) Fallback to handle_conversation (LLM/chatbot) and web search
                bot_resp = ""
                try:
                    bot_resp = handle_conversation(question) or ""
                except Exception as e:
                    st.error(f"Error in handle_conversation: {e}")
                    bot_resp = ""

                if bot_resp.strip():
                    final_answer = bot_resp
                else:
                    web_results = search_web(question)
                    final_answer = "\n\n".join(web_results) if web_results else "No relevant info found."

        # Display final answer
        st.markdown(f"""
            <div style='background:#f2f2f2;padding:10px;border-left:5px solid green;'>
            <b>🤖 Response:</b><br>{final_answer}</div>
        """, unsafe_allow_html=True)

        # Optionally save response and speak (if utils.speak is available)
        try:
            save_text_response(final_answer, st.session_state.get("logged_in_user", "unknown"))
            # speak(final_answer)  # enable if you want TTS on server (beware of blocking)
        except Exception:
            pass

    # --------------------------
    # DAT file search (fixed block)
    # --------------------------
    st.subheader("📂 DAT File Search")
    dat_option = st.checkbox("Enable DAT Search")
    if dat_option:
        dat_file = st.file_uploader("Upload a DAT file", type=["dat"])
        search_segment = st.text_input("Enter known segment (e.g. NM1*87*2)", key="dat_search_segment")
        target_segment_type = st.text_input("Enter target segment (e.g. N3)", key="dat_target_segment")

        if st.button("Search DAT"):
            if dat_file and search_segment and target_segment_type:
                try:
                    dat_content = dat_file.read().decode("utf-8")
                    transactions, current_txn, inside_txn = [], [], False
                    for line in dat_content.splitlines():
                        for seg in line.split("~"):
                            seg = seg.strip()
                            if not seg:
                                continue
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
                except Exception as e:
                    st.error(f"Error processing DAT file: {e}")
            else:
                st.warning("Please upload DAT file and fill both segment fields first.")

    # --------------------------
    # XML search UI (single helper)
    # --------------------------
    st.subheader("🔍 XML Search with Full Context")
    xml_file = st.file_uploader("📂 Upload XML File", type=["xml"], key="xml_search_uploader")

    if xml_file:
        st.success("✅ XML file uploaded successfully!")
        xml_content = xml_file.getvalue()
        source_tag = st.text_input("Enter source tag name (e.g., PolicyNumber):", key="xml_source_tag")
        source_value = st.text_input("Enter source tag value (e.g., H123456789):", key="xml_source_value")
        target_path = st.text_input("Enter target tag/path (optional, e.g., ClaimID, StartDate):", key="xml_target_path")

        if st.button("Search XML"):
            if source_tag and source_value:
                try:
                    results = search_large_xml_bytes(xml_content, source_tag, source_value, target_path)
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

    # Footer: avatar video if exists
    if os.path.exists("A-ZBlueProject/fixed_talking_lady.mp4"):
        st.video("A-ZBlueProject/fixed_talking_lady.mp4")

else:
    st.info("Please log in to use the application.")
