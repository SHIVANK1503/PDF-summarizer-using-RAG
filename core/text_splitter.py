# text_splitter.py

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def split_text_into_chunks(text: str, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_spliter.create_documents([text])
    return chunks
