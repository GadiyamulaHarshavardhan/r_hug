# app/main.py

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from loader import load_documents_from_folder,load_hf_dataset, chunk_documents
from vector_store import get_pg_vectorstore, store_chunks
from rag_engine import query_rag
import uvicorn
import os

app = FastAPI(
    title="RAG FastAPI Service",
    description="Load data into embeddings & query via local LLM",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup vector DB connection
vectorstore = get_pg_vectorstore()

# Endpoint to load all documents from "../data" and embed + store in DB
@app.post("/load-data")
def load_data(dataset: str = "default"):
    docs = load_documents_from_folder("../data", dataset_tag=dataset)
    if not docs:
        return {"status": "No documents found."}
    
    chunks = chunk_documents(docs)
    store_chunks(chunks, vectorstore)
    return {
        "status": "Success",
        "docs_loaded": len(docs),
        "chunks_created": len(chunks),
        "dataset_tag": dataset
    }

# Endpoint to ask a question to the RAG system
@app.get("/ask")
def ask(
    question: str = Query(...),
    dataset: str = Query(None),
    web_search: bool = False
):
    dataset_filter = {"dataset": dataset} if dataset else None
    answer = query_rag(
        question,
        vectorstore,
        dataset_filter=dataset_filter,
        use_web_search=web_search
    )
    return {
        "question": question,
        "answer": answer,
        "dataset_used": dataset or "all",
        "web_search_used": web_search
    }

@app.post("/load-hf")
def load_hf(
    dataset: str = "UrbanSyn/UrbanSyn",
    config: str = None,              # Optional config
    split: str = "train[:1%]",       # Default small split for testing
):
    """
    Load any Hugging Face dataset into the RAG system.
    
    Args:
        dataset: Dataset name on Hugging Face (e.g., 'Med-dataset/Med_Dataset')
        config: Optional config name (e.g., 'en' for allenai/c4)
        split: Data split to load (e.g., 'train[:5%]')
    """
    docs = load_hf_dataset(dataset, config=config, split=split, dataset_tag="hf_dataset")
    chunks = chunk_documents(docs)
    store_chunks(chunks, vectorstore)
    return {
        "status": "Success",
        "dataset": dataset,
        "config": config,
        "split": split,
        "docs_loaded": len(docs),
        "chunks_created": len(chunks),
        "dataset_tag": "hf_dataset"
    }

# Run directly with `python3 main.py`
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
