Android mobile playback patch for Aarvi Voice Document Assistant

1. In VOICE_HTML, immediately after the opening <div id="aarvi-voice-root">, insert:
<div class="aarvi-controls aarvi-mobile-controls">
  <button id="aarvi-mobile-read">🔊 Listen to entire document (mobile)</button>
  <button id="aarvi-mobile-page">🔊 Listen to selected PDF page (mobile)</button>
</div>

2. In VOICE_JS, after the stopReadingButton declaration, insert:
const mobileReadButton = root.querySelector("#aarvi-mobile-read");
const mobilePageButton = root.querySelector("#aarvi-mobile-page");

3. In VOICE_JS, inside the if (!state.handlersReady) block, insert BEFORE state.handlersReady = true:
mobileReadButton.onclick = () => {
    stopRecognition();
    const text = root.__mobileDocumentText || "";
    if (!text.trim()) { setStatus("Upload a readable document first."); return; }
    speakResponse(text);
};
mobilePageButton.onclick = () => {
    stopRecognition();
    const text = root.__mobilePageText || "";
    if (!text.trim()) { setStatus("Select a readable PDF page first."); return; }
    speakResponse(text);
};

4. In VOICE_JS, just after const state = root.__aarviState; insert:
root.__mobileDocumentText = data?.document_text || "";
root.__mobilePageText = data?.selected_page_text || "";

5. In Python voice_component(data={...}), add:
"document_text": full_document_text,
"selected_page_text": pdf_page_texts.get(st.session_state.get("selected_pdf_page"), ""),

6. In VOICE_CSS, add:
#aarvi-voice-root .aarvi-mobile-controls { margin-bottom: 10px; }
#aarvi-mobile-read, #aarvi-mobile-page { background: #146b63; color: white; }

This is a minimal patch. For very long documents, browser speech synthesis may still stop when Android locks or backgrounds the page. The mobile buttons directly invoke speech synthesis in their tap handlers, improving the likelihood of playback. If browser playback remains unreliable, use generated audio files and a native audio player.
