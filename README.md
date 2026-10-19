# RAG-based PDF Summarizer

A Streamlit app that summarises a PDF and answers questions about it using
Retrieval-Augmented Generation (RAG) with Google Gemini.

## How it works

```
PDF upload -> loader -> text_splitter -> vector_store -> rag_chain (Gemini) -> Streamlit UI
```

1. `core/loader.py` extracts text from the uploaded PDF.
2. `core/text_splitter.py` splits the text into overlapping chunks.
3. `core/vector_store.py` embeds the chunks into a Chroma vector store and returns a retriever.
4. `core/rag_chain.py` retrieves relevant chunks and sends them to Gemini to produce the summary or answer.
5. `streamlit/app.py` is the web interface.

## Project structure

```
RAG_based_PDF_summerizer/
├── config/
│   ├── __init__.py
│   └── settings.py        # model names, chunk size/overlap
├── core/
│   ├── __init__.py
│   ├── loader.py
│   ├── text_splitter.py
│   ├── vector_store.py
│   └── rag_chain.py
├── streamlit/
│   └── app.py             # Streamlit entry point
├── .env.example           # template for your API key
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10 or newer
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

## Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd RAG_based_PDF_summerizer
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your API key**

   Copy `.env.example` to `.env` and put in your key:
   ```bash
   # Windows
   copy .env.example .env
   # macOS / Linux
   cp .env.example .env
   ```
   Then edit `.env`. If your `config/settings.py` reads a different variable
   name, use that name instead.

## Run the app

From the project root:

```bash
streamlit run streamlit/app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Usage

1. Upload a text-based PDF in the sidebar.
2. Click **Process PDF** to generate the summary.
3. Ask follow-up questions in the Q&A section.

## Configuration

Change chunk size, overlap and model names in `config/settings.py`.
To retrieve more context per answer, raise `'k'` in
`core/vector_store.py` (`search_kwargs={'k': 1}`).

## Known limitations

- Scanned (image-only) PDFs are not supported because there is no OCR step.
- The PDF is re-indexed for every question, so answers can be slow on large files.

## Security

- Never commit `.env` or any confidential PDFs. Both are listed in `.gitignore`.
- If a key is ever pushed to GitHub, revoke it and create a new one.

## Contributing

1. Fork the repo and create a feature branch.
2. Keep changes focused and test with a sample PDF.
3. Open a pull request describing what you changed and why.
