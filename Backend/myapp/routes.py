#backend/myapp/routes.py
from flask import Flask, Blueprint, request, jsonify
from PyPDF2 import PdfReader
from models.rag_model import RAGModel
import os

routes = Blueprint('routes', __name__)

# Initialize RAG Model with environment variables
cohere_api_key = os.getenv('COHERE_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENV')

rag_model = RAGModel(cohere_api_key, pinecone_api_key, pinecone_env)

@routes.route('/upload', methods=['POST'])
def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'}), 400

        pdf_reader = PdfReader(file)
        document_text = ''
        for page in pdf_reader.pages:
            document_text += page.extract_text()

        # Generate and normalize the embedding
        embeddings = rag_model.embed_documents([document_text])
        normalized_embedding = rag_model.normalize_embedding(embeddings[0]).tolist()

        # Add document to Pinecone
        document_id = "doc_1"  # For demo, use a unique ID in production
        rag_model.index.upsert([(document_id, normalized_embedding, {"text": document_text})])

        return jsonify({'status': 'success', 'embedding': normalized_embedding})
    except Exception as e:
        print(f"Upload Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@routes.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data.get('question', '')

        if not question:
            return jsonify({'status': 'error', 'message': 'No question provided'}), 400

        # Retrieve relevant documents from Pinecone
        matches = rag_model.query(question)
        if 'error' in matches:
            return jsonify({'status': 'error', 'message': matches['error']}), 500

        # Extract context from matched documents
        context = ' '.join([match['metadata']['text'] for match in matches])
        
        # If context is empty, return a meaningful response
        if not context:
            return jsonify({'status': 'error', 'message': 'No relevant document found for the question'}), 404
        
        # Generate an answer using Cohere
        answer = rag_model.generate_answer(context, question)
        print(f"Generated Answer: {answer}")

        # Return the answer if generated, else handle edge cases
        if not answer:
            return jsonify({'status': 'error', 'message': 'Failed to generate an answer'}), 500

        return jsonify({'status': 'success', 'answer': answer})
    except Exception as e:
        print(f"Ask Question Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
