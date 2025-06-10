# app/vector_store.py

import os
import logging
from langchain_community.vectorstores import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load .env before anything else

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_pg_vectorstore():
    db_connection = os.getenv("POSTGRES_CONNECTION", 
        "postgresql://postgres:password@localhost:5432/rag_db"
    )
    logger.info("[VECTORSTORE] Connecting to PostgreSQL...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = PGVector(
        collection_name="rag_data",
        connection_string=db_connection,
        embedding_function=embeddings
    )
    logger.info("[VECTORSTORE] Connection established.")
    return vectorstore


def store_chunks(chunks, vectorstore):
    logger.info(f"[STORE] Storing {len(chunks)} chunks into PostgreSQL...")
    vectorstore.add_documents(chunks)
    logger.info("[STORE] Done.")