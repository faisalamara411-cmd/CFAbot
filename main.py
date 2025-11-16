# main.py
import json
import os
import time
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import numpy as np

st.set_page_config(page_title="CFA Chatbot")

# --------- Load JSONL corpus ---------
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "train.jsonl")

docs = []
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

st.write(f"Loaded {len(docs)} documents.")

# --------- Compute embeddings ---------
st.write("Loading embeddings model (SentenceTransformer)...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
doc_embeddings = embedding_model.encode(docs, convert_to_numpy=True)
st.write("Embeddings ready.")

# --------- Load local LLM for summarization/answering ---------
st.write("Loading LLM model (T5-small)...")
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# --------- Helper functions ---------
def get_top_docs(query, top_k=3):
    query_emb = embedding_model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_emb, doc_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [docs[i] for i in top_indices]

def generate_answer(query):
    top_docs = get_top_docs(query)
    context = " ".join(top_docs)
    prompt = f"Use the following context to answer the question.\nContext: {context}\nQuestion: {query}\nAnswer:"
    output = llm_pipeline(prompt, max_length=256, do_sample=False)
    return output[0]['generated_text']

# --------- Streamlit chat interface ---------
st.title("CFA Chatbot")

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
