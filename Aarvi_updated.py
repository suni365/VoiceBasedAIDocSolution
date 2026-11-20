import streamlit as st
from docx import Document
from pypdf import PdfReader
from io import BytesIO
import re

st.set_page_config(page_title="Voice Document Assistant", page_icon="🎙️", layout="wide")
st.title("🎙️ Voice-Based Intelligent Document Assistant")
st.write("Upload a PDF, Word, or text document. Ask a question or say a keyword. "
         "Ask questions by typing or speaking. Aarvi can listen continuously, answer from your document, and read answers aloud.")

COMMON_QA = {
    "how are you": "I am fine thank you and hoping that you too is having a great day by the way how are you ",
    "I am fine thank you ": "Good to know that god bless you always, you are special and kind ",
    "I am fine": "Good to know that god bless you always",
    "how are you doing": "I am fine. How are you?",
    "what do you do": "I work for a company called Headlthedge, its a part of UST global",
    "Where do you work": "I work for a company called Headlthedge, its a part of UST global",
    "what makes you happy": "I fond of listening music, oberving people, exploring life, I too like doing meditation and yoga",
    "where are you from": "I am from Trivandrum, Kerala, its a southern part of India ",
    "who created you": "I have been created by Sunita Panicker she is a believer in transformation, resilience, and the unseen power that drives human potential. ",
    "who is your creator": "Sunita created me.",
    "who inspires you": "Sunita inspires me with her curiosity, determination and interest in technology.",
    "what is your name": "My name is AI Chatboat Voice-Based Document Assistant ",
    "what can you do": "I can read uploaded documents, find relevant information, answer questions and speak the answers aloud.",
    "thank you": "You are welcome! and Have a great day ahead ",
    "thanks": "You are welcome!"
}

def normalize(text):
    return re.sub(r"[^a-z0-9\s]", " ", str(text).lower()).strip()

def get_common_answer(question):
    q = normalize(question)
    if not q:
        return None
    normalized_answers = {normalize(question): answer for question, answer in COMMON_QA.items()}
    if q in normalized_answers:
        return normalized_answers[q]
    for known_question, answer in normalized_answers.items():
        if known_question and known_question in q and len(q) >= 5:
            return answer
    return None

def split_into_paragraphs(page_text):
    page_text = page_text.replace("\r\n", "\n").replace("\r", "\n")
    raw_paragraphs = re.split(r"\n\s*\n", page_text)
    if len(raw_paragraphs) == 1:
        raw_paragraphs = [line.strip() for line in page_text.splitlines() if line.strip()]
    cleaned = []
    for paragraph in raw_paragraphs:
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if len(paragraph) >= 15:
            cleaned.append(paragraph)
    return cleaned

def extract_document(uploaded_file):
    name = uploaded_file.name.lower()
    chunks = []
    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            for paragraph in split_into_paragraphs(page_text):
                chunks.append({"text": paragraph, "page": page_number})
    elif name.endswith(".docx"):
        doc = Document(BytesIO(uploaded_file.getvalue()))
        for paragraph in doc.paragraphs:
            text = re.sub(r"\s+", " ", paragraph.text).strip()
            if len(text) >= 15:
                chunks.append({"text": text, "page": None})
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(re.sub(r"\s+", " ", cell.text).strip() for cell in row.cells).strip()
                if len(row_text) >= 15:
                    chunks.append({"text": row_text, "page": None})
    elif name.endswith(".txt"):
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        for paragraph in split_into_paragraphs(text):
            chunks.append({"text": paragraph, "page": None})
    else:
        raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")
    return chunks

def extract_full_document(uploaded_file):
    name = uploaded_file.name.lower()
    full_parts = []
    page_texts = {}
    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            page_texts[page_number] = page_text
            if page_text:
                full_parts.append(f"Page {page_number}.\n{page_text}")
    elif name.endswith(".docx"):
        doc = Document(BytesIO(uploaded_file.getvalue()))
        for paragraph in doc.paragraphs:
            paragraph_text = paragraph.text.strip()
            if paragraph_text:
                full_parts.append(paragraph_text)
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells).strip()
                if row_text:
                    full_parts.append(row_text)
    elif name.endswith(".txt"):
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        full_parts.append(text)
    return "\n\n".join(full_parts).strip(), page_texts

STOP_WORDS = {
    "the", "is", "are", "a", "an", "of", "to", "in", "on", "for",
    "and", "or", "what", "where", "when", "who", "how", "can",
    "could", "please", "tell", "me", "about", "find", "show",
    "give", "does", "do", "this", "that", "it", "my", "your",
    "explain", "describe", "would", "should", "which", "with",
    "from", "was", "were", "be", "being", "has", "have", "had"
}

def get_keywords(text):
    words = re.findall(r"[a-z0-9]+", normalize(text))
    return {word for word in words if word not in STOP_WORDS and len(word) > 1}

def find_relevant_paragraphs(question, chunks, max_results=2):
    query = normalize(question)
    query_words = get_keywords(question)
    if not query_words:
        return []
    scored = []
    for chunk in chunks:
        paragraph = chunk["text"]
        paragraph_normalized = normalize(paragraph)
        paragraph_words = get_keywords(paragraph)
        matched = query_words.intersection(paragraph_words)
        if not matched:
            continue
        coverage = len(matched) / len(query_words)
        precision = len(matched) / max(len(paragraph_words), 1)
        score = (0.8 * coverage) + (0.2 * precision)
        if query and query in paragraph_normalized:
            score += 0.5
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] < 0.18:
        return []
    results = []
    seen = set()
    best_score = scored[0][0]
    for score, chunk in scored:
        if score < max(0.18, best_score * 0.55):
            continue
        key = normalize(chunk["text"])
        if key not in seen:
            results.append(chunk)
            seen.add(key)
        if len(results) >= max_results:
            break
    return results

def find_keyword_paragraphs(keyword, chunks, max_results=5):
    keyword_normalized = normalize(keyword)
    if not keyword_normalized:
        return []
    keyword_words = set(keyword_normalized.split())
    matches = []
    for chunk in chunks:
        paragraph_normalized = normalize(chunk["text"])
        paragraph_words = set(paragraph_normalized.split())
        exact_phrase = keyword_normalized in paragraph_normalized
        all_words_present = bool(keyword_words) and keyword_words.issubset(paragraph_words)
        if exact_phrase or all_words_present:
            matches.append(chunk)
    return matches[:max_results]

def display_and_speak(text):
    text = str(text).strip()
    st.session_state["aarvi_response_text"] = text
    st.session_state["aarvi_response_id"] = st.session_state.get("aarvi_response_id", 0) + 1

def process_command(command, mode, chunks):
    command = command.strip()
    if not command:
        st.warning("Please enter or speak a question or keyword.")
        return
    st.markdown(f"**Command:** {command}")
    if mode == "Ask a Question":
        common_answer = get_common_answer(command)
        if common_answer:
            display_and_speak(common_answer)
            return
        if not chunks:
            st.warning("Please upload a document first.")
            return
        results = find_relevant_paragraphs(command, chunks, max_results=2)
    else:
        if not chunks:
            st.warning("Please upload a document first.")
            return
        results = find_keyword_paragraphs(command, chunks, max_results=5)
    if not results:
        display_and_speak("I could not find a relevant paragraph in the uploaded document. "
                          "Please try another question or keyword.")
        return
    if mode == "Keyword Search":
        st.success(f"Found {len(results)} matching paragraph(s) for '{command}'.")
    else:
        st.success("I found the most relevant paragraph(s) in your document.")
    answer_parts = []
    for index, result in enumerate(results, start=1):
        paragraph_text = result["text"]
        if len(results) > 1:
            answer_parts.append(f"Matching paragraph {index}:\n{paragraph_text}")
        else:
            answer_parts.append(paragraph_text)
        if result["page"] is not None:
            st.caption(f"Source: PDF page {result['page']}")
    answer = "\n\n".join(answer_parts)
    display_and_speak(answer)

uploaded_file = st.file_uploader("Upload PDF, Word, or Text document", type=["pdf", "docx", "txt"])
chunks = []
# Layout change: reserve the voice control location before the document details.
voice_controls_container = st.container()
if uploaded_file is not None:
    try:
        chunks = extract_document(uploaded_file)
        if not chunks:
            st.error("No readable paragraphs were found. "
                     "If this is a scanned PDF, it may need OCR.")
        else:
            st.success(f"Document loaded successfully: {len(chunks)} paragraph(s) or text sections found.")
    except Exception as error:
        st.error(f"Could not read the document: {error}")

full_document_text = ""
pdf_page_texts = {}
if uploaded_file is not None:
    try:
        full_document_text, pdf_page_texts = extract_full_document(uploaded_file)
        if full_document_text:
            st.divider()
            st.subheader("📖 Read the uploaded document")
            if st.button("🔊 Read the entire document aloud", key="read_entire_document"):
                display_and_speak(full_document_text)
            if uploaded_file.name.lower().endswith(".pdf") and pdf_page_texts:
                selected_page = st.selectbox("Choose a PDF page to read", options=list(pdf_page_texts.keys()),
                                             format_func=lambda page: f"Page {page}", key="selected_pdf_page")
                if st.button("🔊 Read selected PDF page", key="read_selected_pdf_page"):
                    selected_text = pdf_page_texts.get(selected_page, "").strip()
                    if selected_text:
                        display_and_speak(f"Page {selected_page}.\n{selected_text}")
                    else:
                        st.warning("No selectable text was found on this PDF page. "
                                   "It may be a scanned image and require OCR.")
        else:
            st.info("No readable text was extracted from this document.")
    except Exception as error:
        st.error(f"Could not prepare the full document for reading: {error}")

mode = st.radio("Choose your assistant mode:", ["Ask a Question", "Keyword Search"], horizontal=True)
st.divider()
st.subheader("⌨️ Type a command")
typed_command = st.text_input("Enter your question or keyword:", key="typed_command", placeholder="Example: What is the payment process?")
if st.button("Search typed command", type="primary"):
    process_command(typed_command, mode, chunks)

with voice_controls_container:
    st.divider()
    st.subheader("🎙️ Talk with Aarvi")
    st.write("Press Start conversation once. Speak naturally; Aarvi will process "
             "your question, speak the answer, and then listen for your next question.")

VOICE_HTML = """
<div id="aarvi-voice-root">
    <div class="aarvi-controls">
        <button id="aarvi-start">▶ Start conversation</button>
        <button id="aarvi-stop">■ Stop conversation</button>
    </div>
    <div class="aarvi-controls aarvi-reading-controls">
        <button id="aarvi-pause-reading">⏸ Pause reading</button>
        <button id="aarvi-resume-reading">▶ Resume reading</button>
        <button id="aarvi-stop-reading">■ Stop reading</button>
    </div>
    <p id="aarvi-progress" aria-live="polite">No document is being read.</p>
    <p id="aarvi-status" role="status">
        Press Start conversation and allow microphone access.
    </p>
    <p class="aarvi-hint">
        Continuous turn-taking: Aarvi listens, answers aloud, then listens again.
    </p>
</div>
"""
VOICE_CSS = """
#aarvi-voice-root {
    padding: 12px;
    border: 1px solid var(--st-border-color);
    border-radius: 12px;
    font-family: var(--st-font);
}
#aarvi-voice-root .aarvi-controls {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
#aarvi-voice-root button {
    border: 0;
    border-radius: 8px;
    padding: 10px 16px;
    cursor: pointer;
    font-size: 15px;
}
#aarvi-start {
    background: var(--st-primary-color);
    color: white;
}
#aarvi-stop, #aarvi-pause-reading, #aarvi-resume-reading,
#aarvi-stop-reading {
    background: var(--st-secondary-background-color);
    color: var(--st-text-color);
}
#aarvi-voice-root .aarvi-reading-controls {
    margin-top: 10px;
}
#aarvi-progress {
    margin: 10px 0 4px 0;
    font-size: 0.9rem;
}
#aarvi-status {
    margin: 12px 0 4px 0;
    font-weight: 600;
}
#aarvi-voice-root .aarvi-hint {
    color: var(--st-text-color);
    opacity: 0.7;
    font-size: 0.85rem;
}
"""
VOICE_JS = r"""
export default function ({
    parentElement,
    data,
    setStateValue,
    setTriggerValue
}) {
    const root = parentElement.querySelector("#aarvi-voice-root");
    if (!root) return;

    if (!root.__aarviState) {
        root.__aarviState = {
            recognition: null,
            listening: false,
            waitingForPython: false,
            speaking: false,
            lastResponseId: null,
            restartTimer: null,
            speechChunks: [],
            speechIndex: 0,
            speechSession: 0,
            speechResolve: null,
            speechPaused: false,
            handlersReady: false
        };
    }

    const state = root.__aarviState;
    const startButton = root.querySelector("#aarvi-start");
    const stopButton = root.querySelector("#aarvi-stop");
    const pauseReadingButton = root.querySelector("#aarvi-pause-reading");
    const resumeReadingButton = root.querySelector("#aarvi-resume-reading");
    const stopReadingButton = root.querySelector("#aarvi-stop-reading");
    const status = root.querySelector("#aarvi-status");
    const progress = root.querySelector("#aarvi-progress");

    function setStatus(message) {
        status.textContent = message;
    }

    function stopRecognition() {
        if (state.restartTimer) {
            clearTimeout(state.restartTimer);
            state.restartTimer = null;
        }
        if (state.recognition) {
            const oldRecognition = state.recognition;
            state.recognition = null;
            try {
                oldRecognition.onend = null;
                oldRecognition.stop();
            } catch (error) {}
        }
    }

    function startRecognition() {
        const SpeechRecognition =
            window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            setStatus(
                "Speech recognition is unavailable here. Please use the latest "
                + "Google Chrome or Microsoft Edge."
            );
            state.listening = false;
            setStateValue("listening", false);
            return;
        }

        if (
            state.recognition || state.speaking ||
            state.waitingForPython || !state.listening
        ) return;

        const recognition = new SpeechRecognition();
        state.recognition = recognition;
        recognition.lang = "en-IN";
        recognition.continuous = true;
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        recognition.onstart = () => {
            setStatus("🟢 Listening... Ask Aarvi something.");
        };

        recognition.onresult = (event) => {
            let finalText = "";
            for (let i = event.resultIndex; i < event.results.length; i++) {
                if (event.results[i].isFinal) {
                    finalText += event.results[i][0].transcript + " ";
                }
            }
            finalText = finalText.trim();

            if (finalText) {
                state.waitingForPython = true;
                setStatus("I heard: " + finalText + " — thinking...");
                stopRecognition();
                setTriggerValue("command", {
                    text: finalText,
                    nonce: Date.now()
                });
            }
        };

        recognition.onerror = (event) => {
            if (
                event.error === "not-allowed" ||
                event.error === "service-not-allowed"
            ) {
                state.listening = false;
                state.waitingForPython = false;
                setStateValue("listening", false);
                setStatus(
                    "Microphone permission was blocked. Allow microphone access "
                    + "for this site in Chrome, then press Start again."
                );
            } else if (event.error === "network") {
                setStatus(
                    "Speech recognition needs an internet connection. "
                    + "Trying to reconnect..."
                );
            } else if (
                event.error !== "no-speech" && event.error !== "aborted"
            ) {
                setStatus("Speech recognition issue: " + event.error);
            }
        };

        recognition.onend = () => {
            if (state.recognition === recognition) {
                state.recognition = null;
            }
            if (
                state.listening && !state.speaking &&
                !state.waitingForPython
            ) {
                state.restartTimer = setTimeout(() => {
                    state.restartTimer = null;
                    startRecognition();
                }, 350);
            }
        };

        try {
            recognition.start();
        } catch (error) {
            state.recognition = null;
            setStatus("Could not start the microphone: " + error.message);
        }
    }

    function splitForSpeech(text, maxLength = 1600) {
        const sentences = text.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [text];
        const chunks = [];
        let current = "";

        for (const sentence of sentences) {
            const part = sentence.trim();
            if (!part) continue;
            if ((current + " " + part).trim().length > maxLength && current) {
                chunks.push(current.trim());
                current = part;
            } else {
                current = (current + " " + part).trim();
            }
        }
        if (current) chunks.push(current);
        return chunks;
    }

    function finishSpeech(session) {
        if (session !== state.speechSession) return;
        state.speaking = false;
        state.speechPaused = false;
        state.speechChunks = [];
        state.speechIndex = 0;
        progress.textContent = "Reading finished.";
        const resolve = state.speechResolve;
        state.speechResolve = null;
        if (resolve) resolve();
    }

    function stopSpeech(message = "Reading stopped.") {
        state.speechSession += 1;
        state.speaking = false;
        state.speechPaused = false;
        state.speechChunks = [];
        state.speechIndex = 0;
        if ("speechSynthesis" in window) window.speechSynthesis.cancel();
        progress.textContent = message;
        const resolve = state.speechResolve;
        state.speechResolve = null;
        if (resolve) resolve();
    }

    function speakResponse(text) {
        return new Promise((resolve) => {
            if (!text || !("speechSynthesis" in window)) {
                setStatus("Speech output is not supported. The answer remains visible.");
                resolve();
                return;
            }

            stopSpeech("Preparing reading...");
            const session = state.speechSession;
            state.speechChunks = splitForSpeech(text);
            state.speechIndex = 0;
            state.speechResolve = resolve;
            state.speaking = true;
            state.speechPaused = false;
            setStatus("🔊 Aarvi is speaking...");

            function speakNext() {
                if (session !== state.speechSession || state.speechPaused) return;
                if (state.speechIndex >= state.speechChunks.length) {
                    finishSpeech(session);
                    return;
                }
                progress.textContent = `Reading section ${state.speechIndex + 1} of ${state.speechChunks.length}`;
                const utterance = new SpeechSynthesisUtterance(state.speechChunks[state.speechIndex]);
                utterance.lang = "en-IN";
                utterance.rate = 1;
                utterance.pitch = 1;
                utterance.onend = () => {
                    if (session !== state.speechSession) return;
                    state.speechIndex += 1;
                    speakNext();
                };
                utterance.onerror = (event) => {
                    if (session !== state.speechSession) return;
                    if (event.error === "canceled" || event.error === "interrupted") return;
                    state.speechIndex += 1;
                    speakNext();
                };
                window.speechSynthesis.speak(utterance);
            }
            speakNext();
        });
    }

    function pauseSpeech() {
        if (!state.speaking || state.speechPaused) return;
        state.speechPaused = true;
        if ("speechSynthesis" in window) window.speechSynthesis.pause();
        setStatus("⏸ Reading paused. Press Resume reading to continue.");
    }

    function resumeSpeech() {
        if (!state.speaking || !state.speechPaused) return;
        state.speechPaused = false;
        if ("speechSynthesis" in window) window.speechSynthesis.resume();
        setStatus("🔊 Aarvi is continuing the reading...");
    }

    if (!state.handlersReady) {
        startButton.onclick = () => {
            state.listening = true;
            state.waitingForPython = false;
            setStateValue("listening", true);
            setStatus("Starting microphone...");
        };

        stopButton.onclick = () => {
            state.listening = false;
            state.waitingForPython = false;
            stopRecognition();
            stopSpeech("Reading stopped with the conversation.");
            setStateValue("listening", false);
            setStatus("Conversation stopped. Press Start when ready.");
        };

        pauseReadingButton.onclick = pauseSpeech;
        resumeReadingButton.onclick = resumeSpeech;
        stopReadingButton.onclick = () => {
            stopSpeech("Reading stopped. You can start another reading from the document controls.");
            setStatus("Reading stopped.");
            if (state.listening) startRecognition();
        };
        state.handlersReady = true;
    }

    state.listening = Boolean(data?.listening);
    const responseId = data?.response_id ?? 0;
    const responseText = data?.response_text ?? "";

    if (responseText && responseId !== state.lastResponseId) {
        state.lastResponseId = responseId;
        state.waitingForPython = false;
        stopRecognition();

        speakResponse(responseText).then(() => {
            if (state.listening) {
                setStatus(
                    "Aarvi has finished. Listening for your next question..."
                );
                startRecognition();
            } else {
                setStatus("Aarvi has finished speaking.");
            }
        });
    } else if (!state.listening) {
        stopRecognition();
        if (!state.speaking) setStatus("Conversation is paused.");
    } else if (
        state.listening && !state.speaking && !state.waitingForPython
    ) {
        startRecognition();
    }
}
"""

voice_component = st.components.v2.component(
    name="aarvi_continuous_voice",
    html=VOICE_HTML,
    css=VOICE_CSS,
    js=VOICE_JS
)

def sync_aarvi_listening_state():
    component_state = st.session_state.get("aarvi_voice_component")
    st.session_state["aarvi_listening_enabled"] = bool(
        getattr(component_state, "listening", False)
    )

latest_answer = st.session_state.get("aarvi_response_text", "")
if latest_answer:
    st.markdown("### Aarvi's answer")
    st.write(latest_answer)

# Render the same persistent component in the reserved top container.
with voice_controls_container:
    voice_result = voice_component(
        data={
            "listening": bool(st.session_state.get("aarvi_listening_enabled", False)),
            "response_id": st.session_state.get("aarvi_response_id", 0),
            "response_text": st.session_state.get("aarvi_response_text", "")
        },
        default={"listening": False},
        key="aarvi_voice_component",
        on_listening_change=sync_aarvi_listening_state,
        on_command_change=lambda: None
    )

voice_command = getattr(voice_result, "command", None)
if voice_command and isinstance(voice_command, dict):
    command_nonce = voice_command.get("nonce")
    command_text = voice_command.get("text", "").strip()
    if command_text and command_nonce != st.session_state.get("last_aarvi_command_nonce"):
        st.session_state["last_aarvi_command_nonce"] = command_nonce
        process_command(command_text, mode, chunks)
        st.rerun()

st.markdown("""
<div style="
    margin-top: 40px;
    padding: 18px;
    text-align: center;
    border-top: 1px solid #cccccc;
">
    <div style="font-size: 13px; color: #888;">
        Designed &amp; Created with ❤️ by
    </div>
    <div style="
        font-size: 21px;
        font-weight: 600;
        color: #D4A84B;
        margin-top: 6px;
    ">
        Sunita Kumari Panicker
    </div>
    <div style="font-size: 12px; color: #888;">
        Creator & Developer
    </div>
</div>
""", unsafe_allow_html=True)
