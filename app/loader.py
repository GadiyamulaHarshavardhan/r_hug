# app/loader.py

import logging
from pathlib import Path
from langchain_community.document_loaders import TextLoader, CSVLoader, PyMuPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_dataset  # For HF dataset support

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIN_PAGE_LENGTH = 50  # Minimum number of chars to consider a valid page


def clean_null_bytes(text: str) -> str:
    """Replace null bytes in text."""
    return text.replace("\x00", "")


def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    """Splits documents into smaller chunks and cleans null bytes."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    for chunk in chunks:
        chunk.page_content = clean_null_bytes(chunk.page_content)
    return chunks


def load_documents_from_folder(folder_path: str, dataset_tag: str = "default") -> list[Document]:
    """
    Loads all supported files from folder and subfolders.
    Supported types: .txt, .csv, .pdf
    """
    docs = []
    logger.info(f"[LOADER] Scanning folder recursively: {folder_path}")

    # Supported extensions
    supported_exts = [".txt", ".csv", ".pdf"]

    for path in Path(folder_path).rglob("*"):
        if path.is_file():
            ext = path.suffix.lower()
            logger.info(f"[LOADER] Found file: {path} ({ext})")

            try:
                # Skip unsupported file types
                if ext not in supported_exts:
                    logger.warning(f"[SKIP] Unsupported file type: {ext}")
                    continue

                # File size check (skip tiny/empty files)
                if path.stat().st_size < 100:
                    logger.warning(f"[SKIP] File too small (<100 bytes): {path}")
                    continue

                # Choose loader based on extension
                if ext == ".txt":
                    loader = TextLoader(str(path), encoding="utf-8")
                elif ext == ".csv":
                    loader = CSVLoader(str(path))
                elif ext == ".pdf":
                    loader = PyMuPDFLoader(str(path))

                # Load document(s)
                loaded_docs = loader.load()
                logger.info(f"[LOADER] Loaded {len(loaded_docs)} documents from {path}")

                # Clean and tag each document
                tagged_docs = [
                    Document(
                        page_content=clean_null_bytes(doc.page_content),
                        metadata={**doc.metadata, "source": str(path), "dataset": dataset_tag}
                    )
                    for doc in loaded_docs
                    if len(doc.page_content.strip()) > MIN_PAGE_LENGTH
                ]
                docs.extend(tagged_docs)

            except Exception as e:
                logger.error(f"[ERROR] Failed to load {path}: {e}", exc_info=True)

    logger.info(f"[LOADER] Total documents loaded: {len(docs)}")
    return docs


def load_hf_dataset(dataset_name: str, split: str = "train[:5%]", config: str = None, dataset_tag: str = "hf_data"):
    """
    Loads datasets from HuggingFace Hub.
    """
    ds = load_dataset(dataset_name, config, split=split)

    docs = []
    logger.info(f"[HF LOADER] Loading dataset: {dataset_name}, Config: {config}, Split: {split}")

    for i, item in enumerate(ds):
        text = item.get("text", "") or f"{item.get('question', '')}\n{item.get('answer', '')}"
        if text.strip():
            doc = Document(
                page_content=text,
                metadata={"source": item.get("id", f"item_{i}"), "dataset": dataset_tag}
            )
            docs.append(doc)

    logger.info(f"[HF LOADER] Loaded {len(docs)} documents from HuggingFace dataset.")
    return docs


if __name__ == "__main__":
    import os
    from pprint import pprint

    folder_path = os.path.join("..", "data")
    docs = load_documents_from_folder(folder_path, dataset_tag="sample_data")

    print(f"[TEST] Loaded {len(docs)} documents")

    for i, doc in enumerate(docs[:3]):
        print(f"\n--- Document {i+1} ---")
        pprint(doc.page_content[:500])