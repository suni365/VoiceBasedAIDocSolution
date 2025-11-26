# A-ZChatWithVoice.py
# RAG-enabled (OpenAI embeddings + Chat), audio/video commented out
import os
from io import BytesIO
import streamlit as st
from lxml import etree
import docx
import xml.etree.ElementTree as ET
import numpy as np
import openai
import time

# keep utils imports (they provide auth, helpers, legacy search)
from utils import (
    authenticate_user, clean_text, handle_conversation, search_in_doc,
    search_web, save_text_response, speak, search_excel, search_pdf,
    get_base64_image, AudioProcessor
)

# --------------------------
# Safety: comment out system-level installs (left as comments)
# --------------------------
# os.system("apt-get install -y ffmpeg > /dev/null 2>&1")
# from pydub import AudioSegment
# import speech_recognition as sr
# from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --------------------------
# OpenAI key (robust)
# --------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or (st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None)
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# --------------------------
# Helpers: chunking and text extraction
# --------------------------
def chunk_text(text, chunk_size=400, overlap=50):
    """Chunk by words with overlap. Returns list of strings."""
    if not text:
        return []
    words = text.split()
    if len(words) == 0:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks

def extract_text_from_docx_bytes(b):
    try:
        from tempfile import NamedTemporaryFile
        tmp = NamedTemporaryFile(delete=False, suffix=".docx")
        tmp.write(b)
        tmp.flush()
        tmp.close()
        doc = docx.Document(tmp.name)
        paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(paras)
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass

def extract_text_from_pdf_bytes(b):
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(BytesIO(b))
        pages = []
        for p in reader.pages:
            text = p.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)
    except Exception:
        return ""

def extract_text_from_excel_bytes(b):
    try:
        import pandas as pd
        from tempfile import NamedTemporaryFile
        tmp = NamedTemporaryFile(delete=False, suffix=".xlsx")
        tmp.write(b)
        tmp.flush()
        tmp.close()
        xls = pd.ExcelFile(tmp.name)
        texts = []
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            texts.append(df.to_csv(index=False))
        return "\n".join(texts)
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass

def extract_text_from_dat_bytes(b):
    try:
        return b.decode(errors="ignore")
    except:
        return ""

def extract_text_from_xml_bytes(b):
    try:
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(BytesIO(b), parser)
        return etree.tostring(tree.getroot(), encoding="unicode")
    except Exception:
        return ""

# --------------------------
# Embedding helpers using OpenAI
# --------------------------
def embed_texts_openai(texts, model="text-embedding-3-small"):
    """
    texts: list[str]
    returns: numpy array shape (len(texts), dim)
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)  # placeholder shape if needed
    embs = []
    batch_size = 50  # smaller batches to be safe
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = openai.Embeddings.create(model=model, input=batch)
        for d in resp["data"]:
            embs.append(np.array(d["embedding"], dtype=np.float32))
    return np.vstack(embs)

def normalize(v):
    if v is None or len(v) == 0:
        return v
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms

def cosine_search(query_emb, corpus_emb, top_k=4):
    """Return indices of top_k closest (cosine) and scores"""
    if corpus_emb is None or corpus_emb.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    # If query_emb is shape (dim,) convert to (1,dim)
    q = np.asarray(query_emb)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    # assume both are normalized
    sims = np.dot(corpus_emb, q.T).squeeze()
    if sims.ndim == 0:
        sims = np.array([sims])
    order = np.argsort(-sims)[:top_k]
    return order, sims[order]

# --------------------------
# Streamlit UI config
# --------------------------
st.set_page_config(layout="wide")
st.sidebar.title("Voice-Driven Intelligent Document Assistant")
st.sidebar.image("A-ZBlueProject/AIChatbot.png", use_container_width=True)
st.sidebar.title("🔑 User Authentication")

# session-state for embeddings/chunks
if "rag_chunks" not in st.session_state:
    st.session_state["rag_chunks"] = []       # list of strings
if "rag_embeddings" not in st.session_state:
    st.session_state["rag_embeddings"] = None  # numpy array
if "rag_sources" not in st.session_state:
    st.session_state["rag_sources"] = []      # filenames or metadata

# --------------------------
# Authentication
# --------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state["logged_in_user"] = ""

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
    # Welcome block
    img_base64 = get_base64_image("A-ZBlueProject/sunita.png")
    st.markdown(f"""
        <div style="position:fixed;top:50px;right:10px;background:#333;padding:10px;border-radius:10px;color:white;">
        <img src="data:image/png;base64,{img_base64}" width="100">
        <p><b>Welcome, {st.session_state['logged_in_user']}!</b></p>
        <p style='font-size:12px;color:#ff9800;'>Created by Sunita Panicker</p></div>
    """, unsafe_allow_html=True)

    st.title("🤖 AI Doc Chatbot (OpenAI RAG, Audio Disabled)")

    # --------------------------
    # Excel & PDF Search (legacy helpers retained)
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
    # Document upload (for RAG) and voice upload (commented/kept)
    # --------------------------
    uploaded_files = st.file_uploader("Upload documents for RAG (docx, pdf, xlsx, dat, xml)", accept_multiple_files=True)
    # voice_file = st.file_uploader("Upload a voice file (.m4a/.wav)")   # Disabled for now
    user_input = st.text_input("Ask something (will use uploaded docs):")

    # Buttons & index controls
    with st.sidebar.expander("RAG Settings"):
        chunk_size = st.number_input("Chunk size (words)", min_value=100, max_value=1200, value=400)
        chunk_overlap = st.number_input("Chunk overlap (words)", min_value=0, max_value=400, value=50)
        top_k = st.slider("Top K passages", 1, 8, 4)
        embed_model_name = st.text_input("OpenAI Embedding Model", value="text-embedding-3-small")
        # hint for API
        if not OPENAI_API_KEY:
            st.warning("OPENAI_API_KEY not set. RAG embeddings/generation will not work. App will use keyword fallback.")

    # Process uploads and build/update RAG index
    if uploaded_files:
        new_chunks = []
        new_sources = []
        for f in uploaded_files:
            fname = f.name
            st.write(f"Processing {fname} ...")
            b = f.read()
            text = ""
            if fname.lower().endswith(".docx"):
                text = extract_text_from_docx_bytes(b)
            elif fname.lower().endswith(".pdf"):
                text = extract_text_from_pdf_bytes(b)
            elif fname.lower().endswith(".xlsx") or fname.lower().endswith(".xls"):
                text = extract_text_from_excel_bytes(b)
            elif fname.lower().endswith(".dat"):
                text = extract_text_from_dat_bytes(b)
            elif fname.lower().endswith(".xml"):
                text = extract_text_from_xml_bytes(b)
            else:
                st.warning(f"Unsupported file type: {fname}")
                continue

            # chunk by words using chosen size/overlap
            chunks_local = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
            for c in chunks_local:
                new_chunks.append(c)
                new_sources.append(fname)

        if new_chunks:
            # If OpenAI key present -> build embeddings using OpenAI
            if OPENAI_API_KEY:
                try:
                    st.info("Creating embeddings with OpenAI (this may take a few seconds)...")
                    embs = embed_texts_openai(new_chunks, model=embed_model_name)
                    embs = normalize(embs)
                    # append to session state
                    if st.session_state["rag_embeddings"] is None:
                        st.session_state["rag_embeddings"] = embs
                        st.session_state["rag_chunks"] = new_chunks
                        st.session_state["rag_sources"] = new_sources
                    else:
                        st.session_state["rag_embeddings"] = np.vstack([st.session_state["rag_embeddings"], embs])
                        st.session_state["rag_chunks"].extend(new_chunks)
                        st.session_state["rag_sources"].extend(new_sources)
                    st.success(f"Indexed {len(new_chunks)} new chunks.")
                except Exception as e:
                    st.error(f"Embedding creation failed: {e}\nFalling back to keyword-only indexing.")
                    st.session_state["rag_chunks"].extend(new_chunks)
                    st.session_state["rag_sources"].extend(new_sources)
            else:
                st.session_state["rag_chunks"].extend(new_chunks)
                st.session_state["rag_sources"].extend(new_sources)
                st.success(f"Stored {len(new_chunks)} chunks (keyword fallback mode).")

    # --------------------------
    # Voice processing (disabled/commented)
    # --------------------------
    # if voice_file:
    #     st.write("Processing voice...")
    #     user_input = process_uploaded_voice(voice_file)
    #     st.write(f"**You said:** {user_input}")

    # --------------------------
    # Handle user query
    # --------------------------
    response = None
    if user_input and user_input.strip():
        # RAG path using OpenAI embeddings + chat
        if OPENAI_API_KEY and st.session_state.get("rag_embeddings") is not None and len(st.session_state.get("rag_chunks", [])) > 0:
            try:
                q_emb = embed_texts_openai([user_input], model=embed_model_name)
                q_emb = normalize(q_emb)
                idxs, scores = cosine_search(q_emb.squeeze(), st.session_state["rag_embeddings"], top_k=top_k)
                if idxs.size == 0:
                    response = "No indexed passages to search."
                else:
                    # if best score below threshold => not relevant
                    best_score = float(scores[0]) if scores.size>0 else 0.0
                    if best_score < 0.12:
                        # threshold can be tuned
                        response = "I could not find relevant information in the uploaded documents for that question."
                    else:
                        context_passages = []
                        for idx in idxs:
                            idx = int(idx)
                            context_passages.append(f"Source: {st.session_state['rag_sources'][idx]}\n{st.session_state['rag_chunks'][idx]}")

                        sys_prompt = ("You are a helpful assistant. Answer the user's question using ONLY the provided context passages. "
                                      "Do not use any outside knowledge or hallucinate. If the answer is not in the context, say you couldn't find it.")
                        user_prompt = f"Context:\n\n{'\n\n'.join(context_passages)}\n\nQuestion: {user_input}\n\nProvide a concise answer and mention which source(s) you used."

                        try:
                            chat_resp = openai.ChatCompletion.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role":"system","content":sys_prompt},
                                    {"role":"user","content":user_prompt}
                                ],
                                temperature=0.0,
                                max_tokens=400
                            )
                            response = chat_resp['choices'][0]['message']['content'].strip()
                        except Exception as e:
                            st.error(f"OpenAI ChatCompletion failed: {e}. Falling back to returning top passages.")
                            # fallback: return top passages
                            response = "\n\n".join(context_passages)
            except Exception as e:
                st.error(f"RAG generation error: {e}\nFalling back to keyword search.")
                response = None

        # fallback keyword search (simple substring) or web search
        if (not response or response.strip() == ""):
            if st.session_state.get("rag_chunks"):
                q_lower = user_input.lower()
                matches = []
                for i, chunk in enumerate(st.session_state["rag_chunks"]):
                    if q_lower in chunk.lower():
                        matches.append(f"[{st.session_state['rag_sources'][i]}] {chunk}")
                if matches:
                    response = "\n\n".join(matches[:10])
                else:
                    web = search_web(user_input)
                    response = "\n\n".join(web) if web else "No relevant info found in documents."
            else:
                response = "No documents indexed. Upload documents first or ask a keyword that may match."

    # --------------------------
    # Show response
    # --------------------------
    st.markdown(f"""
        <div style='background:#f2f2f2;padding:10px;border-left:5px solid green;'>
        <b>🤖 Response:</b><br>{response or 'Ask a question after uploading documents.'}</div>
    """, unsafe_allow_html=True)

    # --------------------------
    # Video response (commented out)
    # --------------------------
    # st.video("A-ZBlueProject/fixed_talking_lady.mp4")
    # generate_video_response(response)   # intentionally commented

    # --------------------------
    # DAT and XML sections (kept as-is)
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
# Standalone XML search helpers (kept below)
# --------------------------
def search_large_xml_file(xml_file, source_tag, source_value, target_path=None):
    tree = etree.parse(xml_file)
    root = tree.getroot()
    results = []
    for elem in root.iter(source_tag):
        if elem.text and elem.text.strip() == source_value.strip():
            parent = elem
            while parent.getparent() is not None:
                parent = parent.getparent()
            if target_path:
                for t in parent.iter(target_path):
                    results.append(etree.tostring(t, pretty_print=True, encoding='unicode'))
            else:
                results.append(etree.tostring(parent, pretty_print=True, encoding='unicode'))
    return results

def search_large_xml_bytes(xml_content, source_tag, source_value, target_path=None):
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(BytesIO(xml_content), parser)
    root = tree.getroot()
    results = []
    for elem in root.iter(source_tag):
        if elem.text and elem.text.strip() == source_value.strip():
            parent = elem
            while parent.getparent() is not None:
                parent = parent.getparent()
            if target_path:
                for t in parent.iter(target_path):
                    results.append(etree.tostring(t, pretty_print=True, encoding='unicode'))
            else:
                results.append(etree.tostring(parent, pretty_print=True, encoding='unicode'))
    return results
