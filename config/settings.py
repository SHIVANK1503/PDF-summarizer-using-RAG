from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
EMBEDDING_LLM_MODEL = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-2')
CHAT_LLM_MODEL = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 500
RETRIEVER_K = 1