# vector_store.py

from langchain_community.vectorstores import Chroma
from config.settings import EMBEDDING_LLM_MODEL


def create_retriever_from_chunks(chunks: list, batch_size: int = 1):
    if not chunks:
        raise ValueError('The Provided chunks list is empty.')
    else:
        vector_store = Chroma.from_documents(
            documents=[chunks[0]], embedding=EMBEDDING_LLM_MODEL)
        for i in range(1, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vector_store.add_documents(batch)
        retriver = vector_store.as_retriever(
            search_type='similarity', search_kwargs={'k': 1})
        return retriver
