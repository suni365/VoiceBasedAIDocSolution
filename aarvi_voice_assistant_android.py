import streamlit as st
from docx import Document
from pypdf import PdfReader
from io import BytesIO
import re

st.set_page_config(page_title='Voice Document Assistant', page_icon='🎙️', layout='wide')
st.title('🎙️ Voice-Based Intelligent Document Assistant')
st.write('Upload a PDF, Word, or text document. Ask a question or say a keyword. Aarvi can listen, answer from your document, and read answers aloud.')

COMMON_QA = {
    'how are you': 'I am fine thank you and hoping that you too is having a great day by the way how are you',
    'i am fine thank you': 'Good to know that god bless you always, you are special and kind',
    'i am fine': 'Good to know that god bless you always',
    'how are you doing': 'I am fine. How are you?',
    'what do you do': 'I work for a company called Headlthedge, its a part of UST global',
    'where do you work': 'I work for a company called Headlthedge, its a part of UST global',
    'what makes you happy': 'I fond of listening music, oberving people, exploring life, I too like doing meditation and yoga',
    'where are you from': 'I am from Trivandrum, Kerala, its a southern part of India',
    'who created you': 'I have been created by Sunita Panicker she is a believer in transformation, resilience, and the unseen power that drives human potential.',
    'who is your creator': 'Sunita created me.',
    'who inspires you': 'Sunita inspires me with her curiosity, determination and interest in technology.',
    'what is your name': 'My name is AI Chatboat Voice-Based Document Assistant',
    'what can you do': 'I can read uploaded documents, find relevant information, answer questions and speak the answers aloud.',
    'thank you': 'You are welcome! and Have a great day ahead',
    'thanks': 'You are welcome!'
}

def normalize(text):
    return re.sub(r'[^a-z0-9\s]', ' ', str(text).lower()).strip()

def get_common_answer(question):
    q = normalize(question)
    if not q: return None
    answers = {normalize(k): v for k, v in COMMON_QA.items()}
    if q in answers: return answers[q]
    for known, answer in answers.items():
        if known and known in q and len(q) >= 5: return answer
    return None

def split_into_paragraphs(page_text):
    page_text = page_text.replace('\r\n','\n').replace('\r','\n')
    parts = re.split(r'\n\s*\n', page_text)
    if len(parts) == 1: parts = [line.strip() for line in page_text.splitlines() if line.strip()]
    return [s for p in parts if len(s := re.sub(r'\s+', ' ', p).strip()) >= 15]

def extract_document(uploaded_file):
    name = uploaded_file.name.lower()
    chunks = []
    if name.endswith('.pdf'):
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        for number, page in enumerate(reader.pages, 1):
            for paragraph in split_into_paragraphs(page.extract_text() or ''):
                chunks.append({'text': paragraph, 'page': number})
    elif name.endswith('.docx'):
        doc = Document(BytesIO(uploaded_file.getvalue()))
        for p in doc.paragraphs:
            text = re.sub(r'\s+', ' ', p.text).strip()
            if len(text) >= 15: chunks.append({'text': text, 'page': None})
        for table in doc.tables:
            for row in table.rows:
                text = ' | '.join(re.sub(r'\s+', ' ', cell.text).strip() for cell in row.cells).strip()
                if len(text) >= 15: chunks.append({'text': text, 'page': None})
    elif name.endswith('.txt'):
        text = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        chunks = [{'text': p, 'page': None} for p in split_into_paragraphs(text)]
    else:
        raise ValueError('Unsupported file type. Please upload PDF, DOCX, or TXT.')
    return chunks

def extract_full_document(uploaded_file):
    name = uploaded_file.name.lower()
    parts, pages = [], {}
    if name.endswith('.pdf'):
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        for number, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or '').strip()
            pages[number] = text
            if text: parts.append(f'Page {number}.\n{text}')
    elif name.endswith('.docx'):
        doc = Document(BytesIO(uploaded_file.getvalue()))
        parts.extend(p.text.strip() for p in doc.paragraphs if p.text.strip())
        for table in doc.tables:
            for row in table.rows:
                text = ' | '.join(cell.text.strip() for cell in row.cells).strip()
                if text: parts.append(text)
    elif name.endswith('.txt'):
        parts.append(uploaded_file.getvalue().decode('utf-8', errors='ignore'))
    return '\n\n'.join(parts).strip(), pages

STOP_WORDS = set('the is are a an of to in on for and or what where when who how can could please tell me about find show give does do this that it my your explain describe would should which with from was were be being has have had'.split())
def get_keywords(text):
    return {w for w in re.findall(r'[a-z0-9]+', normalize(text)) if w not in STOP_WORDS and len(w)>1}

def find_relevant_paragraphs(question, chunks, max_results=2):
    query, query_words = normalize(question), get_keywords(question)
    if not query_words: return []
    scored = []
    for chunk in chunks:
        p = chunk['text']; p_norm = normalize(p)
        p_words = get_keywords(p); matched = query_words & p_words
        if not matched: continue
        score = 0.8 * len(matched)/len(query_words) + 0.2 * len(matched)/max(len(p_words),1)
        if query and query in p_norm: score += 0.5
        scored.append((score, chunk))
    scored.sort(key=lambda x:x[0], reverse=True)
    if not scored or scored[0][0] < 0.18: return []
    best, results, seen = scored[0][0], [], set()
    for score, chunk in scored:
        if score < max(0.18, best*0.55): continue
        key = normalize(chunk['text'])
        if key not in seen: results.append(chunk); seen.add(key)
        if len(results) >= max_results: break
    return results

def find_keyword_paragraphs(keyword, chunks, max_results=5):
    phrase = normalize(keyword)
    if not phrase: return []
    words, matches = set(phrase.split()), []
    for chunk in chunks:
        p = normalize(chunk['text'])
        if phrase in p or words.issubset(set(p.split())): matches.append(chunk)
    return matches[:max_results]

def display_and_speak(text):
    st.session_state['aarvi_response_text'] = str(text).strip()
    st.session_state['aarvi_response_id'] = st.session_state.get('aarvi_response_id',0)+1

def process_command(command, mode, chunks):
    command = command.strip()
    if not command:
        st.warning('Please enter or speak a question or keyword.'); return
    st.markdown(f'**Command:** {command}')
    if mode == 'Ask a Question':
        common = get_common_answer(command)
        if common: display_and_speak(common); return
    if not chunks:
        st.warning('Please upload a document first.'); return
    results = (find_relevant_paragraphs(command, chunks) if mode == 'Ask a Question'
               else find_keyword_paragraphs(command, chunks))
    if not results:
        display_and_speak('I could not find a relevant paragraph in the uploaded document. Please try another question or keyword.'); return
    if mode == 'Keyword Search': st.success(f"Found {len(results)} matching paragraph(s) for '{command}'.")
    else: st.success('I found the most relevant paragraph(s) in your document.')
    parts=[]
    for i, result in enumerate(results,1):
        p=result['text']; parts.append(f'Matching paragraph {i}:\n{p}' if len(results)>1 else p)
        if result['page'] is not None: st.caption(f"Source: PDF page {result['page']}")
    display_and_speak('\n\n'.join(parts))

uploaded_file = st.file_uploader('Upload PDF, Word, or Text document', type=['pdf','docx','txt'])
chunks=[]
voice_controls_container=st.container()
if uploaded_file is not None:
    try:
        chunks=extract_document(uploaded_file)
        if chunks: st.success(f'Document loaded successfully: {len(chunks)} paragraph(s) or text sections found.')
        else: st.error('No readable paragraphs were found. If this is a scanned PDF, it may need OCR.')
    except Exception as error: st.error(f'Could not read the document: {error}')

full_document_text=''; pdf_page_texts={}
if uploaded_file is not None:
    try:
        full_document_text,pdf_page_texts=extract_full_document(uploaded_file)
        if full_document_text:
            st.divider(); st.subheader('📖 Read the uploaded document')
            st.caption('On Android, use the mobile Listen button in the voice controls above. The original buttons remain available.')
            if st.button('🔊 Read the entire document aloud', key='read_entire_document'):
                display_and_speak(full_document_text)
            if uploaded_file.name.lower().endswith('.pdf') and pdf_page_texts:
                selected_page=st.selectbox('Choose a PDF page to read', options=list(pdf_page_texts), format_func=lambda p:f'Page {p}', key='selected_pdf_page')
                if st.button('🔊 Read selected PDF page', key='read_selected_pdf_page'):
                    text=pdf_page_texts.get(selected_page,'').strip()
                    if text: display_and_speak(f'Page {selected_page}.\n{text}')
                    else: st.warning('No selectable text was found on this PDF page. It may require OCR.')
        else: st.info('No readable text was extracted from this document.')
    except Exception as error: st.error(f'Could not prepare the full document for reading: {error}')

mode=st.radio('Choose your assistant mode:', ['Ask a Question','Keyword Search'], horizontal=True)
st.divider(); st.subheader('⌨️ Type a command')
typed_command=st.text_input('Enter your question or keyword:',key='typed_command',placeholder='Example: What is the payment process?')
if st.button('Search typed command',type='primary'): process_command(typed_command,mode,chunks)

with voice_controls_container:
    st.divider(); st.subheader('🎙️ Talk with Aarvi')
    st.write('Press Start conversation once. Speak naturally; Aarvi will process your question, speak the answer, and then listen for your next question.')

VOICE_HTML = '''
<div id="aarvi-voice-root">
  <div class="aarvi-controls aarvi-mobile-controls">
    <button id="aarvi-mobile-read">🔊 Listen to entire document (mobile)</button>
    <button id="aarvi-mobile-page">🔊 Listen to selected PDF page (mobile)</button>
  </div>
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
  <p id="aarvi-status" role="status">Press Start conversation and allow microphone access.</p>
  <p class="aarvi-hint">Continuous turn-taking: Aarvi listens, answers aloud, then listens again.</p>
</div>
'''
VOICE_CSS = '''
#aarvi-voice-root { padding:12px; border:1px solid var(--st-border-color); border-radius:12px; font-family:var(--st-font); }
#aarvi-voice-root .aarvi-controls { display:flex; gap:10px; flex-wrap:wrap; }
#aarvi-voice-root button { border:0; border-radius:8px; padding:10px 16px; cursor:pointer; font-size:15px; min-height:44px; }
#aarvi-start { background:var(--st-primary-color); color:white; }
#aarvi-stop, #aarvi-pause-reading, #aarvi-resume-reading, #aarvi-stop-reading { background:var(--st-secondary-background-color); color:var(--st-text-color); }
#aarvi-voice-root .aarvi-reading-controls { margin-top:10px; }
#aarvi-voice-root .aarvi-mobile-controls { margin-bottom:10px; }
#aarvi-mobile-read, #aarvi-mobile-page { background:#146b63; color:white; }
#aarvi-progress { margin:10px 0 4px; font-size:.9rem; }
#aarvi-status { margin:12px 0 4px; font-weight:600; }
#aarvi-voice-root .aarvi-hint { color:var(--st-text-color); opacity:.7; font-size:.85rem; }
'''
VOICE_JS = r'''
export default function ({ parentElement, data, setStateValue, setTriggerValue }) {
    const root = parentElement.querySelector('#aarvi-voice-root');
    if (!root) return;
    if (!root.__aarviState) {
        root.__aarviState = {
            recognition:null, listening:false, waitingForPython:false, speaking:false,
            lastResponseId:null, restartTimer:null, speechChunks:[], speechIndex:0,
            speechSession:0, speechResolve:null, speechPaused:false, handlersReady:false,
            activeUtterance:null
        };
    }
    const state = root.__aarviState;
    root.__mobileDocumentText = data?.document_text || '';
    root.__mobilePageText = data?.selected_page_text || '';
    const startButton = root.querySelector('#aarvi-start');
    const stopButton = root.querySelector('#aarvi-stop');
    const pauseReadingButton = root.querySelector('#aarvi-pause-reading');
    const resumeReadingButton = root.querySelector('#aarvi-resume-reading');
    const stopReadingButton = root.querySelector('#aarvi-stop-reading');
    const mobileReadButton = root.querySelector('#aarvi-mobile-read');
    const mobilePageButton = root.querySelector('#aarvi-mobile-page');
    const status = root.querySelector('#aarvi-status');
    const progress = root.querySelector('#aarvi-progress');
    function setStatus(message) { status.textContent = message; }
    function stopRecognition() {
        if (state.restartTimer) { clearTimeout(state.restartTimer); state.restartTimer=null; }
        if (state.recognition) {
            const old = state.recognition; state.recognition=null;
            try { old.onend=null; old.stop(); } catch(error) {}
        }
    }
    function startRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            setStatus('Speech recognition is unavailable here. Please use Google Chrome or Microsoft Edge.');
            state.listening=false; setStateValue('listening',false); return;
        }
        if (state.recognition || state.speaking || state.waitingForPython || !state.listening) return;
        const recognition=new SpeechRecognition(); state.recognition=recognition;
        recognition.lang='en-IN'; recognition.continuous=true;
        recognition.interimResults=false; recognition.maxAlternatives=1;
        recognition.onstart=()=>setStatus('🟢 Listening... Ask Aarvi something.');
        recognition.onresult=(event)=>{
            let finalText='';
            for(let i=event.resultIndex;i<event.results.length;i++) {
                if(event.results[i].isFinal) finalText+=event.results[i][0].transcript+' ';
            }
            finalText=finalText.trim();
            if(finalText) {
                state.waitingForPython=true;
                setStatus('I heard: '+finalText+' — thinking...');
                stopRecognition();
                setTriggerValue('command',{text:finalText,nonce:Date.now()});
            }
        };
        recognition.onerror=(event)=>{
            if(event.error==='not-allowed'||event.error==='service-not-allowed') {
                state.listening=false; state.waitingForPython=false;
                setStateValue('listening',false);
                setStatus('Microphone permission blocked. Allow microphone access in Chrome, then press Start.');
            } else if(event.error==='network') {
                setStatus('Speech recognition needs internet. Trying to reconnect...');
            } else if(event.error!=='no-speech'&&event.error!=='aborted') {
                setStatus('Speech recognition issue: '+event.error);
            }
        };
        recognition.onend=()=>{
            if(state.recognition===recognition) state.recognition=null;
            if(state.listening&&!state.speaking&&!state.waitingForPython) {
                state.restartTimer=setTimeout(()=>{state.restartTimer=null;startRecognition();},350);
            }
        };
        try { recognition.start(); }
        catch(error) { state.recognition=null;setStatus('Could not start microphone: '+error.message); }
    }
    // Shorter chunks improve reliability on mobile browsers.
    function splitForSpeech(text,maxLength=220) {
        const sentences=text.match(/[^.!?]+[.!?]+|[^.!?]+$/g)||[text];
        const chunks=[];let current='';
        for(const sentence of sentences) {
            const part=sentence.trim();if(!part)continue;
            if((current+' '+part).trim().length>maxLength && current) {chunks.push(current.trim());current='';}
            if(part.length>maxLength) {
                if(current) {chunks.push(current.trim());current='';}
                const words=part.split(/\s+/);let segment='';
                for(const word of words) {
                    if((segment+' '+word).trim().length>maxLength && segment) {chunks.push(segment);segment=word;}
                    else segment=(segment+' '+word).trim();
                }
                current=segment;
            } else current=(current+' '+part).trim();
        }
        if(current)chunks.push(current);
        return chunks;
    }
    function finishSpeech(session) {
        if(session!==state.speechSession)return;
        state.speaking=false;state.speechPaused=false;state.speechChunks=[];
        state.speechIndex=0;state.activeUtterance=null;
        progress.textContent='Reading finished.';
        const resolve=state.speechResolve;state.speechResolve=null;if(resolve)resolve();
    }
    function stopSpeech(message='Reading stopped.') {
        state.speechSession+=1;state.speaking=false;state.speechPaused=false;
        state.speechChunks=[];state.speechIndex=0;state.activeUtterance=null;
        if('speechSynthesis' in window)window.speechSynthesis.cancel();
        progress.textContent=message;
        const resolve=state.speechResolve;state.speechResolve=null;if(resolve)resolve();
    }
    function speakResponse(text) {
        return new Promise(resolve=>{
            if(!text||!('speechSynthesis' in window)) {
                setStatus('Speech output not supported. The answer remains visible.');resolve();return;
            }
            stopSpeech('Preparing reading...');
            const session=state.speechSession;
            state.speechChunks=splitForSpeech(text);state.speechIndex=0;
            state.speechResolve=resolve;state.speaking=true;state.speechPaused=false;
            setStatus('🔊 Aarvi is speaking...');
            function speakNext() {
                if(session!==state.speechSession||state.speechPaused)return;
                if(state.speechIndex>=state.speechChunks.length) {finishSpeech(session);return;}
                progress.textContent=`Reading section ${state.speechIndex+1} of ${state.speechChunks.length}`;
                const utterance=new SpeechSynthesisUtterance(state.speechChunks[state.speechIndex]);
                state.activeUtterance=utterance;
                utterance.lang='en-IN';utterance.rate=1;utterance.pitch=1;
                utterance.onend=()=>{
                    if(session!==state.speechSession)return;
                    state.speechIndex+=1;speakNext();
                };
                utterance.onerror=(event)=>{
                    if(session!==state.speechSession)return;
                    if(event.error==='canceled'||event.error==='interrupted')return;
                    state.speechIndex+=1;speakNext();
                };
                window.speechSynthesis.speak(utterance);
            }
            speakNext();
        });
    }
    function pauseSpeech() {
        if(!state.speaking||state.speechPaused)return;
        state.speechPaused=true;
        if('speechSynthesis' in window)window.speechSynthesis.pause();
        setStatus('⏸ Reading paused. Press Resume reading to continue.');
    }
    function resumeSpeech() {
        if(!state.speaking||!state.speechPaused)return;
        state.speechPaused=false;
        if('speechSynthesis' in window)window.speechSynthesis.resume();
        setStatus('🔊 Aarvi is continuing the reading...');
    }
    if(!state.handlersReady) {
        startButton.onclick=()=>{
            state.listening=true;state.waitingForPython=false;
            setStateValue('listening',true);setStatus('Starting microphone...');
            startRecognition();
        };
        stopButton.onclick=()=>{
            state.listening=false;state.waitingForPython=false;
            stopRecognition();stopSpeech('Reading stopped with the conversation.');
            setStateValue('listening',false);
            setStatus('Conversation stopped. Press Start when ready.');
        };
        pauseReadingButton.onclick=pauseSpeech;
        resumeReadingButton.onclick=resumeSpeech;
        stopReadingButton.onclick=()=>{
            stopSpeech('Reading stopped. You can start another reading from the document controls.');
            setStatus('Reading stopped.');if(state.listening)startRecognition();
        };
        // Direct click-to-speak is important for Android autoplay restrictions.
        mobileReadButton.onclick=()=>{
            stopRecognition();
            const text=root.__mobileDocumentText||'';
            if(!text.trim()) {setStatus('Upload a readable document first.');return;}
            speakResponse(text);
        };
        mobilePageButton.onclick=()=>{
            stopRecognition();
            const text=root.__mobilePageText||'';
            if(!text.trim()) {setStatus('Select a readable PDF page first.');return;}
            speakResponse(text);
        };
        state.handlersReady=true;
    }
    state.listening=Boolean(data?.listening);
    const responseId=data?.response_id??0;
    const responseText=data?.response_text??'';
    if(responseText&&responseId!==state.lastResponseId) {
        state.lastResponseId=responseId;state.waitingForPython=false;stopRecognition();
        speakResponse(responseText).then(()=>{
            if(state.listening) {
                setStatus('Aarvi has finished. Listening for your next question...');
                startRecognition();
            } else setStatus('Aarvi has finished speaking.');
        });
    } else if(!state.listening) {
        stopRecognition();if(!state.speaking)setStatus('Conversation is paused.');
    } else if(state.listening&&!state.speaking&&!state.waitingForPython)startRecognition();
}
'''

voice_component=st.components.v2.component(name='aarvi_continuous_voice',html=VOICE_HTML,css=VOICE_CSS,js=VOICE_JS)
def sync_aarvi_listening_state():
    component_state=st.session_state.get('aarvi_voice_component')
    st.session_state['aarvi_listening_enabled']=bool(getattr(component_state,'listening',False))

latest_answer=st.session_state.get('aarvi_response_text','')
if latest_answer:
    st.markdown("### Aarvi's answer")
    st.write(latest_answer)

with voice_controls_container:
    voice_result=voice_component(
        data={
            'listening':bool(st.session_state.get('aarvi_listening_enabled',False)),
            'response_id':st.session_state.get('aarvi_response_id',0),
            'response_text':st.session_state.get('aarvi_response_text',''),
            'document_text':full_document_text,
            'selected_page_text':pdf_page_texts.get(st.session_state.get('selected_pdf_page'),''),
        },
        default={'listening':False},key='aarvi_voice_component',
        on_listening_change=sync_aarvi_listening_state,
        on_command_change=lambda:None
    )
voice_command=getattr(voice_result,'command',None)
if voice_command and isinstance(voice_command,dict):
    nonce=voice_command.get('nonce');command_text=voice_command.get('text','').strip()
    if command_text and nonce!=st.session_state.get('last_aarvi_command_nonce'):
        st.session_state['last_aarvi_command_nonce']=nonce
        process_command(command_text,mode,chunks)
        st.rerun()

st.markdown('''
<div style="margin-top:40px;padding:18px;text-align:center;border-top:1px solid #cccccc;">
<div style="font-size:13px;color:#888;">Designed &amp; Created with ❤️ by</div>
<div style="font-size:21px;font-weight:600;color:#D4A84B;margin-top:6px;">Sunita Kumari Panicker</div>
<div style="font-size:12px;color:#888;">Creator &amp; Developer</div>
</div>
''',unsafe_allow_html=True)
