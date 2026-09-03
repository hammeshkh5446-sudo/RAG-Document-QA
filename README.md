# RAG Document Q&A

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions based on the document content.

The application extracts text from PDF files, creates embeddings using Google's Gemini embedding model, retrieves relevant information, and generates answers grounded in the uploaded document.

## Features

- Upload PDF documents
- Automatic text extraction
- Document text chunking
- Gemini-powered embeddings
- Semantic similarity search
- Context-based AI answers
- Source excerpt references
- Streamlit-based user interface

## Tech Stack

- Python
- Streamlit
- Google Gemini API
- Gemini Embeddings
- NumPy
- PyPDF

## How It Works

1. Upload a PDF document.
2. Extract text from the document.
3. Split the content into smaller searchable chunks.
4. Convert document chunks into embeddings.
5. Retrieve the most relevant sections based on the user's question.
6. Generate an AI response using Gemini based only on the retrieved context.

## Installation

Clone the repository:

```bash
git clone https://github.com/hammeshkh5446-sudo/RAG-Document-QA.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## API Key Setup

Create the following file:

```
.streamlit/secrets.toml
```

Add your Gemini API key:

```toml
GEMINI_API_KEY="your_api_key_here"
```

## Run Application

Start the Streamlit application:

```bash
streamlit run app.py
```

## Project Structure

```
RAG-Document-QA/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
    └── secrets.toml
```

## Security

API keys are stored using Streamlit secrets and excluded from Git tracking using `.gitignore`.

## License

MIT License