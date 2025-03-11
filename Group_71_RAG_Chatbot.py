import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from rank_bm25 import BM25Okapi
import pickle  # Replace pickle5 with pickle
import os

# Preprocessing (run locally once, then upload preprocessed data)
def preprocess_pdfs(pdf_paths, max_pages=5):
    import pdfplumber  # Import here to avoid unnecessary dependency if not used
    text_chunks = []
    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= max_pages:  # Limit to first 5 pages per PDF
                    break
                text = page.extract_text()
                if text:
                    for j in range(0, len(text), 300):  # Smaller chunk size
                        chunk = text[j:j+300]
                        text_chunks.append(chunk)
    return text_chunks[:50]  # Limit to 50 chunks total

# Precompute and save (run locally)
def precompute_and_save():
    pdf_paths = ["tesla_10k_2023.pdf", "tesla_10k_2024.pdf"]
    chunks = preprocess_pdfs(pdf_paths)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks, convert_to_tensor=False)
    tokenized_chunks = [chunk.split() for chunk in chunks]
    
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    with open("tokenized_chunks.pkl", "wb") as f:
        pickle.dump(tokenized_chunks, f)

# Uncomment and run locally once, then comment out
# if not os.path.exists("embeddings.pkl"):
#     precompute_and_save()

# Load precomputed data
@st.cache_data
def load_precomputed_data():
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    with open("tokenized_chunks.pkl", "rb") as f:
        tokenized_chunks = pickle.load(f)
    return chunks, embeddings, tokenized_chunks

# Initialize models and index
@st.cache_resource
def load_models_and_index():
    chunks, embeddings, tokenized_chunks = load_precomputed_data()
    
    # FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # BM25
    bm25 = BM25Okapi(tokenized_chunks)
    
    # Smaller SLM
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
    model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m')
    
    # Embedding model (loaded only for queries)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    return chunks, embedder, index, bm25, tokenizer, model

chunks, embedder, index, bm25, tokenizer, model = load_models_and_index()

# Advanced RAG with Multi-Stage Retrieval
def advanced_rag(query):
    # Stage 1: Coarse retrieval with BM25
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    coarse_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
    coarse_chunks = [chunks[i] for i in coarse_indices]
    
    # Stage 2: Fine retrieval with embeddings
    coarse_embeddings = embedder.encode(coarse_chunks)
    coarse_index = faiss.IndexFlatL2(embedder.get_sentence_embedding_dimension())
    coarse_index.add(coarse_embeddings)
    query_embedding = embedder.encode([query])
    D, I = coarse_index.search(query_embedding, k=3)
    final_chunks = [coarse_chunks[i] for i in I[0]]
    
    # Generate response
    context = " ".join(final_chunks)
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    # Use max_new_tokens instead of max_length
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)  # Generate up to 50 new tokens
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Confidence score
    confidence = (bm25_scores[coarse_indices[0]] / max(bm25_scores)) * 0.5 + (1 - D[0][0]) * 0.5
    return answer, confidence

# Guardrail
def guardrail_filter(answer):
    financial_keywords = ["revenue", "profit", "loss", "income", "expense", "balance"]
    if not any(keyword in answer.lower() for keyword in financial_keywords):
        return "Sorry, I couldn’t provide a relevant financial answer."
    return answer

# Streamlit UI
st.title("Financial RAG Chatbot - Group 71")
query = st.text_input("Ask a financial question:")
if query:
    with st.spinner("Processing..."):
        answer, confidence = advanced_rag(query)
        filtered_answer = guardrail_filter(answer)
        st.write(f"Answer: {filtered_answer}")
        st.write(f"Confidence Score: {confidence:.2f}")
