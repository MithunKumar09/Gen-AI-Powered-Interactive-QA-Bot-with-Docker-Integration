# Gen-Ai-RAG-Enhanced-Interactive-QA-Bot

## Overview

**Gen-Ai-RAG-Enhanced-Interactive-QA-Bot** is an advanced interactive question-answering system leveraging Retrieval-Augmented Generation (RAG) with enhanced AI technologies. The system utilizes Cohere for language models and Pinecone for vector search to deliver accurate and dynamic responses to user queries.

## Table of Contents

- [Features]
- [Installation]
- [Configuration]
- [Usage]
- [Directory Structure]
- [Docker Setup]
- [Contributing]
- [License]

## Features

- **Interactive QA System:** Advanced AI models for generating responses.
- **Retrieval-Augmented Generation (RAG):** Integrates retrieval with generation for enhanced accuracy.
- **Cohere Integration:** Utilizes Cohere’s language models for text generation.
- **Pinecone Integration:** Employs Pinecone for efficient vector search and retrieval.
- **Data Storage:** Saves generated answers to the data folder for further processing and analysis.
- **Docker Support:** Streamlined setup and deployment with Docker.
- **Multi-Environment Configuration:** Separate configurations for backend and frontend.

## Installation

### Prerequisites

- **Node.js:** Required for the frontend.
- **Python:** Required for the backend and RAG model.
- **Docker:** For containerized setup.

### Backend Setup

**Navigate to the backend directory**
- cd "Backend (RAG Model)"
- Install the necessary Python packages:

- pip install -r requirements.txt
- Set up the environment variables in the .env file as described in the Configuration section.

### Frontend Setup
**Navigate to the frontend directory:**
- cd "Frontend (Interactive QA Bot)"
- Install the necessary Node.js packages:

- npm install

// *Set up the environment variables in the .env file as described in the Configuration section.*

## Configuration
### Environment Variables
**Create and configure the following .env files:**

- *Backend (Backend (RAG Model)/.env):* Contains configuration for the RAG model, Cohere API key, Pinecone API key, and other backend settings.
- *Frontend (Frontend (Interactive QA Bot)/.env):* Contains configuration for the frontend application, including API URLs.
- Ensure these files are not included in version control. Check .gitignore for details.

### .env Example
- **Backend .env**
- env
- DATABASE_URL=your_database_url
- SECRET_KEY=your_secret_key
- COHERE_API_KEY=your_cohere_api_key
- PINECONE_API_KEY=your_pinecone_api_key
- DATA_FOLDER=path_to_data_folder

**Frontend .env**
- env
- BACKEND_URL=http://backend:5000

## Usage
### Running Locally
**Start the Backend:**
- cd "Backend (RAG Model)"
- python app.py

**Start the Frontend:**
- cd "Frontend (Interactive QA Bot)"
- npm start

### Docker Setup
**Build Docker Images:**
- docker-compose build

**Run Docker Containers:**
- docker-compose up

## Directory Structure
**GEN-AI-RAG-ENHANCED-INTERACTIVE-QA-BOT**
- ├── **Backend (RAG Model)**
- │   ├── pycache
- │   ├── config
- │   ├── data
- │   │   ├── answer_20240917_161421.txt
- │   │   └── answer_20240918_083208.txt
- │   ├── models
- │   ├── myapp
- │   │   ├── pycache
- │   │   ├── __init__.py
- │   │   ├── routes.py
- │   ├── vector_store
- │   │   ├── pycache
- │   │   └── pinecone_db.py
- │   ├── .env
- │   ├── Dockerfile
- │   ├── manage_index.py
- │   └── requirements.txt
- ├── **Frontend (Interactive QA Bot)**
- │   ├── app
- │   │   ├── app.py
- │   ├── static
- │   ├── .env
- │   ├── Dockerfile
- │   └── requirements.txt
- ├── .gitignore
- ├── .dockerignore
- └── docker-compose.yml

## Contributing
**Contributions are welcome! Please follow these steps to contribute:**

- *Fork the Repository*
- *Create a New Branch*
- *Make Your Changes*
- *Commit Your Changes*
- *Push to Your Fork*
- *Create a Pull Request*

### Clone the Repository

- git clone https://github.com/MithunKumar09/Gen-Ai-RAG-Enhanced-Interactive-QA-Bot.git
- cd Gen-Ai-RAG-Enhanced-Interactive-QA-Bot
