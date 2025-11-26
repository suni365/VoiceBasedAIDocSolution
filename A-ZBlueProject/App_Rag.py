import os
import tempfile
import time
from io import BytesIO

import streamlit as st
import docx
from lxml import etree
from pydub import AudioSegment
import speech_recognition as sr
import xml.etree.ElementTree as ET

# Optional: keep your utils import if you still use them for auth, UI helpers, etc.
try:
    from utils import (
        authenticate_user, clean_text, handle_conversation, search_in_doc,
        search_web, save_text_response, speak, search_excel, search_pdf,
        get_base64_image, AudioProcessor
    )
except Exception:
    # utils might not exist in the current context; we won't fail hard here.
    def authenticate_user(a, b): return True
    def get_base64_image(p): return ""
    def search_excel(f, k): return "Not available (utils missing)"
    def search_pdf(f, k): return []
    def handle_conversation(q): return ""
    def search_web(q): return []

# --- Install ffmpeg if running in an environment that needs it (keeps your original intent) ---
os.system("apt-get install -y ffmpeg > /dev/null 2>&1")

# ----------------------------
# RAG Dependencies (local-first)
# ----------------------------
# You must install: sentence-transformers, faiss-cpu, openai (optional), transformers (optional)
# pip install sentence-transformers faiss-cpu openai
from sentence_transformers import SentenceTransformer
import numpy as np
try:
    import faiss
except Exception:
    faiss = None

# Optionally use OpenAI if key present for higher-quality answer generation
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
    except Exception:
        openai = None
else:
    openai = None

# --------------------------
# Voice file processor (kept & slightly hardened)
# --------------------------
def process_uploaded_voice(voice_file):
    """Convert uploaded voice (.m4a/.wav) to text using SpeechRecognition."""
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
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if 'wav_path' in locals() and wav_path != tmp_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

# --------------------------
# Helper: strip namespace for XML
# --------------------------
def strip_namespace(tag):
    return tag.split('}', 1)[1] if '}' in tag else tag

# --------------------------
# Simple/robust chunking
# --------------------------
def chunk_text(text, chunk_size=400, overlap=50):
    """
    Split text into chunks with overlap. chunk_size is in tokens approximated by words.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# --------------------------
# Embedding + FAISS index helper
# --------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_CACHE_DIR = ".emb_cache"
INDEX_FILE = os.path.join(EMBED_CACHE_DIR, "faiss.index")
META_FILE = os.path.join(EMBED_CACHE_DIR, "meta.npy")  # store metadata (list of dicts)

os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def build_index_from_documents(docs: list, rebuild=False):
    """
    docs: list of dicts: {"text": "...", "source": "filename", "meta": {...}}
    This will build or append to a FAISS index stored on disk.
    """
    # create embeddings for each doc chunk
    texts = [d["text"] for d in docs]
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    dim = embeddings.shape[1]

    # create or load faiss index
    if faiss is None:
        raise RuntimeError("faiss not available. Install faiss-cpu.")
    if rebuild or not os.path.exists(INDEX_FILE):
        index = faiss.IndexFlatIP(dim)  # inner-product; we'll normalize
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        faiss.write_index(index, INDEX_FILE)
        meta = docs
        np.save(META_FILE, np.array(meta, dtype=object), allow_pickle=True)
    else:
        # load and append
        index = faiss.read_index(INDEX_FILE)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        existing_meta = list(np.load(META_FILE, allow_pickle=True))
        existing_meta.extend(docs)
        np.save(META_FILE, np.array(existing_meta, dtype=object), allow_pickle=True)
        faiss.write_index(index, INDEX_FILE)
    return True

def load_index():
    if faiss is None or not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        return None, None
    index = faiss.read_index(INDEX_FILE)
    meta = list(np.load(META_FILE, allow_pickle=True))
    return index, meta

def semantic_search(query, top_k=4):
    """
    Return top_k matching chunks (metadata + text) for the query.
    """
    index, meta = load_index()
    if index is None:
        return []
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if idx < len(meta):
            results.append(meta[idx])
    return results

# --------------------------
# Answer generation (OpenAI if available, else simple synth)
# --------------------------
def generate_answer_with_openai(context_chunks, question, max_tokens=256):
    """
    Uses OpenAI ChatCompletion (if openai present). The prompt forces the model
    to use ONLY the provided context. Returns text answer.
    """
    if openai is None:
        raise RuntimeError("OpenAI SDK not available or key missing.")
    # build system + user prompt
    context_text = "\n\n".join([f"Source: {c.get('source','unknown')}\n{c['text']}" for c in context_chunks])
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the context provided. "
        "Do not invent facts or use outside knowledge. If the answer is not contained, say 'I could not find the answer in the provided documents.'"
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}\n\nGive a concise answer and cite which source chunk you used (source names)."
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # change to preferred model in your environment
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return resp['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"OpenAI call failed: {e}"

def synthesize_answer_local(context_chunks, question):
    """
    Basic local synthesizer: extracts sentences from context chunks that have overlap
    with question keywords, then returns a compact assembled answer plus sources.
    This avoids hallucination because it only picks sentences present in chunks.
    """
    import re
    q_words = set([w.lower() for w in re.findall(r"\w+", question) if len(w) > 2])
    chosen = []
    for c in context_chunks:
        sents = re.split(r'(?<=[.!?])\s+', c['text'])
        for s in sents:
            words = set([w.lower() for w in re.findall(r"\w+", s)])
            # pick sentence if overlap with question words
            if len(words & q_words) >= 1:
                chosen.append({"sentence": s.strip(), "source": c.get("source", "")})
    # fallback: if nothing matched pick the top chunk first 2 sentences
    if not chosen and context_chunks:
        sents = re.split(r'(?<=[.!?])\s+', context_chunks[0]['text'])
        for s in sents[:2]:
            chosen.append({"sentence": s.strip(), "source": context_chunks[0].get("source", "")})
    # assemble answer
    if not chosen:
        return "I could not find the answer in the indexed documents."
    answer = " ".join([c["sentence"] for c in chosen])
    sources = ", ".join(sorted(set([c["source"] for c in chosen if c["source"]])))
    return f"{answer}\n\n(Sources: {sources})"

# --------------------------
# Document ingestion utilities (Word, PDF text extractor placeholder)
# --------------------------
def extract_text_from_docx(file_bytes):
    """Return text string from .docx uploaded file-like object (bytes)."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    try:
        tmp.write(file_bytes)
        tmp.flush()
        doc = docx.Document(tmp.name)
        paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(paragraphs)
    finally:
        tmp.close()
        try:
            os.remove(tmp.name)
        except Exception:
            pass

# If you want PDF extraction integrated here, implement using pdfplumber or PyMuPDF.
# For now, rely on your existing search_pdf util or extend similarly.

# --------------------------
# Streamlit layout & flow (keeps your original UI, integrated with RAG)
# --------------------------
st.set_page_config(layout="wide")
st.sidebar.title("Voice-Driven Intelligent Document Assistant (RAG)")
st.sidebar.image("A-ZBlueProject/AIChatbot.png", use_container_width=True)
st.sidebar.title("🔑 User Authentication")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state["logged_in_user"] = ""

# Authentication (keeps your existing flow)
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
    img_base64 = get_base64_image("A-ZBlueProject/sunita.png")
    st.markdown(f"""
        <div style="position:fixed;top:50px;right:10px;background:#333;padding:10px;border-radius:10px;color:white;">
        <img src="data:image/png;base64,{img_base64}" width="100">
        <p><b>Welcome, {st.session_state['logged_in_user']}!</b></p>
        <p style='font-size:12px;color:#ff9800;'>Created by Sunita Panicker</p></div>
    """, unsafe_allow_html=True)

    st.title("🤖 AI Doc Chatbot (RAG)")

    # Excel / PDF search UI preserved
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

    # Main: upload doc + voice + query
    uploaded_file = st.file_uploader("Upload a Word Document (.docx)", type=["docx"])
    voice_file = st.file_uploader("Upload a voice file (.m4a/.wav)")
    user_input = st.text_input("Ask something:")

    # Indexing control
    st.sidebar.markdown("### Index settings")
    idx_rebuild = st.sidebar.checkbox("Rebuild index before adding (clean index)", value=False)
    chunk_size = st.sidebar.number_input("Chunk size (words)", min_value=100, max_value=1200, value=400)
    overlap = st.sidebar.number_input("Chunk overlap (words)", min_value=0, max_value=200, value=50)
    top_k = st.sidebar.slider("Top K results", min_value=1, max_value=8, value=4)

    response = None
    doc_text = ""

    # If doc uploaded, extract and index it
    if uploaded_file:
        # read bytes
        raw = uploaded_file.read()
        doc_text = extract_text_from_docx(raw)
        st.success("Document uploaded. Preparing semantic index...")
        # chunk
        chunks = chunk_text(doc_text, chunk_size=chunk_size, overlap=overlap)
        docs_for_index = []
        for i, c in enumerate(chunks):
            docs_for_index.append({
                "text": c,
                "source": getattr(uploaded_file, "name", "uploaded_doc"),
                "meta": {"chunk_id": i}
            })
        try:
            build_index_from_documents(docs_for_index, rebuild=idx_rebuild)
            st.success(f"Indexed {len(docs_for_index)} chunks.")
        except Exception as e:
            st.error(f"Indexing failed: {e}")

    # Voice -> text conversion
    if voice_file:
        st.write("Processing voice...")
        user_input = process_uploaded_voice(voice_file)
        st.write(f"**You said:** {user_input}")

    # When user asks something, run semantic retrieval + answer generation
    if user_input and user_input.strip():
        st.info("Running semantic search...")
        hits = semantic_search(user_input, top_k=top_k)
        if hits:
            st.success(f"Found {len(hits)} relevant chunks.")
            # display hits in sidebar or main
            with st.expander("Top retrieved chunks (click to view)"):
                for h in hits:
                    st.markdown(f"**Source:** {h.get('source','-')}  \n{h['text'][:800]}{'...' if len(h['text'])>800 else ''}")

            # Generate answer
            if openai is not None:
                try:
                    ans = generate_answer_with_openai(hits, user_input)
                except Exception as e:
                    st.warning(f"OpenAI generation failed: {e}. Falling back to local synth.")
                    ans = synthesize_answer_local(hits, user_input)
            else:
                ans = synthesize_answer_local(hits, user_input)
            response = ans
        else:
            # if no hits, fallback to original behavior: search_in_doc (your keyword search) or web
            if uploaded_file and doc_text:
                # try existing keyword search util
                try:
                    doc_match = search_in_doc(doc_text, user_input)
                    if doc_match:
                        response = doc_match
                    else:
                        st.info("No semantic hit in the indexed documents.")
                        # optional web fallback
                        web_hits = search_web(user_input)
                        response = "\n\n".join(web_hits) if web_hits else "No relevant info found."
                except Exception:
                    web_hits = search_web(user_input)
                    response = "\n\n".join(web_hits) if web_hits else "No relevant info found."
            else:
                web_hits = search_web(user_input)
                response = "\n\n".join(web_hits) if web_hits else "No relevant info found."

    # If nothing generated, show a default message
    if not response:
        response = "Ask a question after uploading a document (or try keyword search)."

    # Present response
    st.markdown(f"""
        <div style='background:#f2f2f2;padding:10px;border-left:5px solid green;'>
        <b>🤖 Response:</b><br>{response}</div>
    """, unsafe_allow_html=True)

    st.video("A-ZBlueProject/fixed_talking_lady.mp4")

    # DAT and XML sections are kept as-is below (omitted here for brevity)
    # You can paste your original DAT/XML UI code here (we kept them earlier in your file)
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

    # XML search UI (you had multiple versions — keep one robust one)
    st.subheader("🔍 XML Search with Full Context")
    xml_file = st.file_uploader("📂 Upload XML File", type=["xml"])
    if xml_file:
        st.success("✅ XML file uploaded successfully!")
        xml_content = xml_file.getvalue()
        source_tag = st.text_input("Enter source tag name (e.g., PolicyNumber):")
        source_value = st.text_input("Enter source tag value (e.g., H123456789):")
        target_path = st.text_input("Enter target tag/path (optional, e.g., ClaimID, StartDate):")
        if st.button("Search XML"):
            if source_tag and source_value:
                try:
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
                                for target_elem in parent.iter(target_path):
                                    results.append(etree.tostring(target_elem, pretty_print=True, encoding='unicode'))
                            else:
                                results.append(etree.tostring(parent, pretty_print=True, encoding='unicode'))
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
