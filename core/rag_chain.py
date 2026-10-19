# rag_chain

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from config.settings import CHAT_LLM_MODEL
from core.loader import extract_text_from_pdf
from core.text_splitter import split_text_into_chunks
from core.vector_store import create_retriever_from_chunks


def format_docs(docs: list) -> str:
    """Helper function to combine retrieved document chunks into a single text block."""
    return '\n\n'.join((doc.page_content for doc in docs))


def prompting():
    prompt = PromptTemplate(template='You are an elite document analysis assistant. Create a highly professional, structured summary of the uploaded document based ONLY on the provided context.\n\nContext from PDF:\n{context}\n\nQuestion: {question}', input_variables=[
                            'context', 'question'])

    def dynamic_rag(input_data):
        raw_bytes = input_data['file_bytes']
        query_text = str(input_data['question'])
        print('-> Running step 1: Extracting text from PDF bytes...')
        raw_text_string = extract_text_from_pdf(raw_bytes)
        print('-> Running step 2: Slicing text into chunk lists...')
        document_chunks = split_text_into_chunks(raw_text_string)
        print('-> Running step 3: Compiling vector database retriever...')
        retriever = create_retriever_from_chunks(document_chunks)
        print('-> Running step 4: Querying your vector database context...')
        retrieved_fragments = retriever.invoke(query_text)
        formatted_context = format_docs(retrieved_fragments)
        final_generation_chain = prompt | CHAT_LLM_MODEL | StrOutputParser()
        print('-> Running step 5: Sending formatted context to Gemini for summary...')
        return final_generation_chain.invoke({'context': formatted_context, 'question': query_text})
    return RunnableLambda(dynamic_rag)


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    load_dotenv()
    path = 'C:\\Users\\Shivank\\Desktop\\RAG\\1_Document_loaders\\dl-curriculum.pdf'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            sample_bytes = f.read()
        print('Starting full LCEL Cross-File Pipeline...')
        master_chain = prompting()
        result = master_chain.invoke(
            {'file_bytes': sample_bytes, 'question': 'Provide an executive summary of this curriculum.'})
        print('\n--- Final Summary Result ---')
        print(result)
    else:
        print(f'Please check your path, test document not found at: {path}')
