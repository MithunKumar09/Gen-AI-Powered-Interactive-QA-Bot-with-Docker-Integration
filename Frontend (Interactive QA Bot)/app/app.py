#frontend/app/app.py
import streamlit as st
import requests
from dotenv import load_dotenv
import os

load_dotenv()

backend_url = os.getenv("BACKEND_URL")

st.title("Interactive QA Bot with RAG")
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    if uploaded_file.size > 0:
        files = {'file': uploaded_file}
        response = requests.post(f"{backend_url}/upload", files=files)
        if response.status_code == 200:
            st.write("Document uploaded successfully")
        else:
            st.write(f"Error uploading document: {response.text}")
    else:
        st.write("Uploaded file is empty. Please upload a different file.")

    question = st.text_input("Ask a question about the document:")
    if question:
        response = requests.post(f"{backend_url}/ask", json={"question": question})
        if response.status_code == 200:
            answer = response.json().get("answer")
            st.write(f"Answer: {answer}")
        else:
            st.write(f"Error fetching answer: {response.text}")
