import streamlit as st




def read_excel(file):
    df = pd.read_excel(file)
    return df.to_string()




def read_dat(file):
    return file.read().decode(errors="ignore")




def read_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()
    return ET.tostring(root, encoding='unicode')


# =======================
# STREAMLIT UI
# =======================


st.title("A-Z Chatbot (RAG Enhanced, Audio Disabled)")


uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)


if uploaded_files:
    all_text_chunks = []


for file in uploaded_files:
    ext = file.name.lower()
    st.write(f"Processing {file.name}...")


if ext.endswith(".docx"):
    text = read_docx(file)
elif ext.endswith(".pdf"):
    text = read_pdf(file)
elif ext.endswith(".xlsx"):
    text = read_excel(file)
elif ext.endswith(".dat"):
    text = read_dat(file)
elif ext.endswith(".xml"):
    text = read_xml(file)
else:
    st.warning(f"Unsupported file type: {file.name}")
    continue


# Chunking
for chunk in text.split(". "):
    if len(chunk.strip()) > 40:
        all_text_chunks.append(chunk.strip())


# Build RAG Index
if all_text_chunks:
    build_rag_index(all_text_chunks)
    st.success("RAG index built successfully! You can now ask natural-language questions.")
    question = st.text_input("Ask anything based on uploaded documents:")

if question:
    response = rag_query(question)
    st.write("### Response:")
    st.write(response)


# Commented audio response
# speak(response)
# generate_video_response(response)
