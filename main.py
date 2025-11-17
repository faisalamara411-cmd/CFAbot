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

docs = []
with st.spinner("Loading documents..."):
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                instruction = item.get("instruction")
                content = item.get("output")
                text = instruction + "\n" + content
                docs.append(text)
            except json.JSONDecodeError:
                continue

# ----------------------- Compute embeddings -----------------------
@st.cache_data
def compute_embeddings(docs):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode(docs, convert_to_numpy=True), model

with st.spinner("Computing embeddings..."):
    doc_embeddings, embedding_model = compute_embeddings(docs)

# ----------------------- Load LLM -----------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return pipe

with st.spinner("Loading LLM model..."):
    llm_pipeline = load_llm()

st.success("CFA Chatbot is ready! Ask me anything about CFA or CIPM programs.")

# ----------------------- Helper functions -----------------------
def get_top_docs(query, top_k=5):
    query_emb = embedding_model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_emb, doc_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [docs[i] for i in top_indices]

def generate_answer(query, top_k=5):
    top_docs = get_top_docs(query, top_k=top_k)
    context = " ".join(top_docs)
    prompt = (
        "You are a CFA exam expert. "
        "Answer the following question based ONLY on the context provided. "
        "If the answer is not in the context, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    output = llm_pipeline(prompt, max_length=256, do_sample=False)
    return output[0]["generated_text"].strip()

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
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = generate_answer(user_input)
        response_placeholder.write(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
