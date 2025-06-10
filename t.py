# app/t.py

from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pdf_loader():
    folder_path = "../data"
    logger.info(f"[TEST] Scanning folder: {folder_path}")

    for path in Path(folder_path).rglob("*"):
        if path.is_file():
            logger.info(f"[FOUND] File: {path} (type: {path.suffix})")
            try:
                if path.suffix == ".pdf":
                    loader = PyMuPDFLoader(str(path))  # Load PDF
                    docs = loader.load()
                    logger.info(f"[LOADED] Loaded {len(docs)} pages from {path}")
                else:
                    logger.warning(f"[SKIP] Not a PDF: {path}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to load {path}: {e}")

if __name__ == "__main__":
    test_pdf_loader()