# main.py
import json
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import numpy as np

# ----------------------- Streamlit page config -----------------------
st.set_page_config(page_title="CFA Chatbot")

# ----------------------- Title -----------------------
st.title("CFA Chatbot")

# ----------------------- Load JSONL corpus -----------------------
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "train.jsonl")

# Function to split long documents into chunks
def chunk_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

docs = []
with st.spinner("Loading documents and splitting into chunks..."):
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                instruction = item.get("instruction", "")
                content = item.get("output", "")
                text = instruction + "\n" + content
                chunks = chunk_text(text)
                docs.extend(chunks)
            except json.JSONDecodeError:
                continue

#st.write(f"Loaded {len(docs)} document chunks.")

# ----------------------- Compute embeddings -----------------------
@st.cache_data
def compute_embeddings(docs):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(docs, convert_to_numpy=True)
    return embeddings, model

with st.spinner("Computing embeddings..."):
    doc_embeddings, embedding_model = compute_embeddings(docs)

# ----------------------- Load instruction-tuned LLM -----------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return pipe

with st.spinner("Loading LLM model..."):
    llm_pipeline = load_llm()

st.success("CFA Chatbot is ready! Ask me anything about CFA or CIPM programs.")

# ----------------------- Helper functions -----------------------
def get_top_docs(query, top_k=5):
    """Retrieve the top_k most relevant documents based on cosine similarity."""
    query_emb = embedding_model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_emb, doc_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [docs[i] for i in top_indices]

def generate_answer(query, top_k=5):
    """Generate an answer using the top documents and the LLM."""
    top_docs = get_top_docs(query, top_k=top_k)
    
    if not top_docs or all(len(doc.strip()) == 0 for doc in top_docs):
        return "I don't know."
    
    context = " ".join(top_docs)
    
    # Strict and simple prompt
    prompt = (
        f"Answer the question using ONLY the following context. "
        f"If the answer is not in the context, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    
    output = llm_pipeline(prompt, max_new_tokens=256, do_sample=False)
    answer = output[0]["generated_text"].strip()
    return answer

# ----------------------- Streamlit chat interface -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about CFA or CIPM programs."}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input
if user_input := st.chat_input("Type a message"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Debug: show retrieved docs (optional)
    # top_docs = get_top_docs(user_input)
    # st.write("Top retrieved document chunks:", top_docs)
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = generate_answer(user_input)
        response_placeholder.write(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

