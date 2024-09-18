# RAG Model with Interactive QA Bot

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) model integrated with a question-answering bot. It includes a backend for handling document uploads and queries, and a frontend for user interactions. The backend utilizes Cohere for embedding and generation, and Pinecone for vector storage. The frontend is built with Streamlit for an interactive user experience.

## Folder Structure

Backend (RAG Model)/ │ ├── myapp/ │ ├── init.py │ └── routes.py │ ├── models/ │ └── rag_model.py │ ├── vector_store/ │ └── pinecone_db.py │ └── newenv/ └── (virtual environment files)

Copy code
Frontend (Interactive QA Bot)/ │ └── app/ └── app.py

## Backend Setup

### Dependencies

Ensure you have the following environment variables set up in your `.env` file:

- `COHERE_API_KEY`: Your API key for Cohere.
- `PINECONE_API_KEY`: Your API key for Pinecone.
- `PINECONE_ENV`: The Pinecone environment (e.g., 'us-west1-gcp').
- `BACKEND_URL`: The URL of your Flask backend (e.g., 'http:// 127.0.0.1:5000').

### Installation

1. **Create and activate a virtual environment:**

    ```bash
    python -m venv newenv
    source newenv/bin/activate  # On Windows use newenv\Scripts\activate
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Flask app:**

    ```bash
    export FLASK_APP=myapp:create_app  # On Windows use: $env:FLASK_APP="myapp:create_app"

    flask run
    ```

### File Descriptions

- `models/rag_model.py`: Contains the implementation of the RAG model, including methods for embedding documents, retrieving documents from Pinecone, generating answers, and saving answers to files.
- `vector_store/pinecone_db.py`: Includes functions to create an index in Pinecone, add documents, and query the index.
- `myapp/routes.py`: Defines the routes for uploading documents and asking questions, interacting with the RAG model.

## Frontend Setup

### Dependenciess

Ensure you have the following environment variable set up in your `.env` file:

- `BACKEND_URL`: The URL of your Flask backend (e.g., 'http:// 127.0.0.1:5000').

### Installations

1. **Create and activate a virtual environment:**

    ```bash
    python -m venv newenv
    source newenv/bin/activate  # On Windows use: newenv\Scripts\activate
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app:**

    ```bash
    streamlit run app/app.py
    ```

## Usage

1. **Upload a PDF Document:**
   - Navigate to the Streamlit app running on `http://localhost:8501`.
   - Use the file uploader to upload a PDF document.
   - The document will be processed, and its embedding will be stored in Pinecone.

2. **Ask a Question:**
   - Enter a question related to the uploaded document.
   - The backend will retrieve relevant documents and generate an answer, which will be displayed on the frontend.

## Notes

- The provided code assumes a basic setup and may need modifications for production use.
- For production deployment, consider using a WSGI server (e.g., Gunicorn) for the Flask app.
