# app/rag_engine.py

from hf_client import query_hf_llm  # Switched from ollama_client

def query_rag(question, vectorstore, dataset_filter=None):
    retrieved_docs = vectorstore.similarity_search(question, k=5, filter=dataset_filter)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:
"""
    return query_hf_llm(prompt)  # Or ollama, depending on what you're using