import os
import re
import base64
import numpy as np
from io import BytesIO
import streamlit as st # Added st for cache_resource
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- NEW IMPORTS for Open Source LLM ---
try:
    import torch
    # ADD BitsAndBytesConfig to your imports
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig 
    # Define a default model. Mistral 7B is popular and capable for RAG.
    # NOTE: You will need sufficient RAM/VRAM to run this model.
    LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
except ImportError:
    LLM_MODEL_NAME = None # Fallback or error handling
# ----------------------------------------

# --------------------------
# 🧠 LLM/Chatbot Interaction (Open Source - Mistral/Hugging Face)
# --------------------------

@st.cache_resource(show_spinner=False)
def load_llm_model():
    """Loads the Hugging Face model and tokenizer once."""
    if not LLM_MODEL_NAME:
        return None, None
        
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

        # -----------------------------------------------------------------
        # ✨ NEW CODE BLOCK: Define the BitsAndBytesConfig
        # This replaces the deprecated torch_dtype and load_in_4bit arguments
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # Set the compute dtype to match your desired torch_dtype
            # This is the correct way to set the compute type for 4-bit models
            bnb_4bit_compute_dtype=torch.bfloat16,
            # You can also add bnb_4bit_quant_type='nf4' and bnb_4bit_use_double_quant=True
        )
        # -----------------------------------------------------------------

        # Load the model with 4-bit quantization for lower VRAM usage
        # You may need to adjust the device mapping based on your environment
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            device_map="auto",
            # REMOVED: torch_dtype=torch.bfloat16, 
            # REMOVED: load_in_4bit=True 
            quantization_config=bnb_config, # <-- NEW ARGUMENT
        )

        # Create the pipeline for easy generation
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            # You can keep torch_dtype here for the pipeline's internal operations
            torch_dtype=torch.bfloat16, 
            device_map="auto",
        )
        return llm_pipeline, tokenizer
        
    except Exception as e:
        print(f"Error loading LLM model {LLM_MODEL_NAME}: {e}")
        return None, None


def handle_conversation(prompt, context=None):
    """
    Interacts with the Hugging Face LLM (Mistral) to generate a response.
    Uses the context (from RAG) to answer the prompt naturally.
    """
    
    llm_pipeline, tokenizer = load_llm_model()

    if llm_pipeline is None:
        return f"LLM Connection Error: Could not load the open-source model '{LLM_MODEL_NAME}'. Check your GPU/RAM/library installation."
        
    # Define the base system instruction/context for the LLM
    system_instruction = (
        "You are an expert document analysis assistant and chatbot. "
        "Your goal is to be helpful and accurate."
    )
    
    # 1. Define the RAG prompt template for the Mistral Instruct model
    if context:
        # Template for RAG (Retrieval-Augmented Generation)
        # This format strongly encourages the model to use the context.
        # Mistral uses a specific instruction format (INST/<<SYS>>).
        rag_instruction = (
            f"<<SYS>>{system_instruction} Your primary task is to answer the user's question using ONLY the provided document CONTEXT. "
            f"Synthesize and rephrase the information into a smooth, natural, and direct answer. Do not just copy the source text. "
            f"If the context does not contain the answer, state politely that the information is not found in the documents. <</SYS>>\n\n"
            f"CONTEXT:\n---\n{context}\n---\n\n"
            f"USER QUESTION: {prompt}\n\n"
            f"Please generate a natural language response based on the CONTEXT that directly answers the USER QUESTION."
        )
        full_prompt = rag_instruction
    else:
        # Template for General Chatbot conversation
        full_prompt = (
            f"<<SYS>>{system_instruction} If no context is provided, respond conversationally as a general chatbot. <</SYS>>\n\n"
            f"{prompt}"
        )
        
    # The final prompt for the model
    messages = [
        {"role": "user", "content": full_prompt}
    ]
    
    # Apply the official chat template
    prompt_with_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        # 2. Generate the response using the pipeline
        response = llm_pipeline(
            prompt_with_template,
            max_new_tokens=512,           # Limit the response length
            do_sample=True,               # Use sampling for creativity
            temperature=0.7,              # Control randomness
            top_p=0.9,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id # Important for batching, set to EOS for single prompt
        )
        
        # Extract the generated text and remove the prompt repetition
        generated_text = response[0]['generated_text']
        
        # Clean up the output to remove the input prompt and instruction template
        # Find where the actual response starts after the user prompt
        if full_prompt in generated_text:
            response_text = generated_text.split(full_prompt, 1)[-1].strip()
        else:
             # Fallback cleanup
             response_text = generated_text
        
        return response_text.strip()

    except Exception as e:
        return f"LLM Generation Error: Could not generate response using the model. Details: {e}"


# --------------------------
# 🔐 Authentication
# ... (all other functions remain the same)
# --------------------------

def authenticate_user(username, password):
# ... (rest of the function is unchanged)
    """Simple authentication."""
    valid_users = {"sunita": "password123"}
    return valid_users.get(username) == password


# --------------------------
# 🧹 Text Cleaning
# ... (rest of the function is unchanged)
# --------------------------
def clean_text(text):
    """Lowercase, strip, remove extra spaces."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


# --------------------------
# 📄 Keyword-based Document Search
# ... (rest of the function is unchanged)
# --------------------------
def search_in_doc(doc_text, keyword):
    """Return sentences containing the keyword."""
    keyword_lower = keyword.lower()
    sentences = re.split(r'(?<=[.!?]) +', doc_text)
    results = [s for s in sentences if keyword_lower in s.lower()]
    return "\n".join(results) if results else None


# --------------------------
# 🌐 Web Search Fallback (Simple Mock)
# ... (rest of the function is unchanged)
# --------------------------
def search_web(query):
    return [
        f"Result 1 for '{query}'",
        f"Result 2 for '{query}'",
        f"Result 3 for '{query}'"
    ]


# --------------------------
# 📝 Save Text Response
# ... (rest of the function is unchanged)
# --------------------------
def save_text_response(filename, text):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)


# --------------------------
# 🔉 Text → Speech (GTTS)
# ... (rest of the function is unchanged)
# --------------------------
def speak(text, filename="response.mp3"):
    try:
        from gtts import gTTS
        tts = gTTS(text)
        tts.save(filename)
        return filename
    except Exception:
        return None


# --------------------------
# 📊 Excel Search
# ... (rest of the function is unchanged)
# --------------------------
def search_excel(excel_file, keyword):
    try:
        import pandas as pd
        xls = pd.ExcelFile(excel_file)
        all_matches = []

        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            df_str = df.astype(str)
            mask = df_str.apply(
                lambda row: row.str.contains(keyword, case=False, na=False)
            ).any(axis=1)
            matches = df[mask]
            if not matches.empty:
                all_matches.append(matches)

        return pd.concat(all_matches) if all_matches else pd.DataFrame()

    except Exception as e:
        return str(e)


# --------------------------
# 📕 PDF Search
# ... (rest of the function is unchanged)
# --------------------------
def search_pdf(pdf_file, keyword):
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(pdf_file)
        results = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and keyword.lower() in text.lower():
                for line in text.splitlines():
                    if keyword.lower() in line.lower():
                        results.append((i + 1, line.strip()))

        return results

    except Exception:
        return []


# --------------------------
# 🖼️ Base64 Image Helper
# ... (rest of the function is unchanged)
# --------------------------
def get_base64_image(img_path):
    if not os.path.exists(img_path):
        return ""
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return encoded


# --------------------------
# 🔉 AudioProcessor Placeholder
# ... (rest of the class is unchanged)
# --------------------------
class AudioProcessor:
    @staticmethod
    def process(file):
        return "Processed text from audio"


# --------------------------
# 🧠 OpenAI RAG Embeddings
# ... (This section is now unused as your main.py uses SentenceTransformer)
# --------------------------
def embed_texts_openai(texts, model="text-embedding-3-small"):
    # ... (function remains for completeness but is not called in main.py)
    return np.zeros((0, 1536)) # Return dummy for now


def normalize(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def cosine_search(query_emb, corpus_emb, top_k=4):
    sims = np.dot(corpus_emb, query_emb.T).squeeze()
    top_idx = np.argsort(-sims)[:top_k]
    top_scores = sims[top_idx]
    return top_idx, top_scores


# --------------------------
# 🎤 Voice File Processor
# ... (rest of the function is unchanged)
# --------------------------
def process_uploaded_voice(voice_file):
    # ... (function remains unchanged)
    pass


# --------------------------
# 🧾 XML Utilities
# ... (rest of the functions are unchanged)
# --------------------------
def strip_namespace(tag):
# ... (function remains unchanged)
    return tag.split('}', 1)[1] if '}' in tag else tag


def search_large_xml_bytes(xml_content, source_tag, source_value, target_path=None):
# ... (function remains unchanged)
    pass

