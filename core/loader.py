# loader.py

import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name
        loader = PyPDFLoader(temp_path)
        document_data = loader.lazy_load()
        document_content = ' '.join(
            (page.page_content for page in document_data))
        return document_content
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
