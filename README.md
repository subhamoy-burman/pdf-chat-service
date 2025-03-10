# Local Vector Store with LangChain and Faiss

This repository demonstrates how to use LangChain to load PDF documents, create local vector stores using Faiss, and perform retrieval-augmented generation (RAG) with OpenAI embeddings.

## Table of Contents

- Introduction
- Setup
- Usage
- Contributing
- License

## Introduction

This project showcases the use of LangChain to load and process PDF documents, create local vector stores using Faiss, and perform retrieval-augmented generation (RAG) with OpenAI embeddings. The goal is to enable efficient local processing and similarity search without relying on cloud-based vector stores.

## Setup

1. **Clone the repository:**

2. **Create a virtual environment:**
    ```bash
    pipenv shell
    ```

3. **Install the required packages:**
    ```bash
    pip install langchain pypdf langchain-openai langchain-community langchainhub faiss-cpu
    ```

## Usage

1. **Download a PDF document:**
    - Download the ReAct paper or any other PDF document and place it in the repository directory.

2. **Run the main script:**
    - Open `main.py` in your preferred IDE (VSCode or PyCharm).
    - Populate the OpenAI API key in the script.
    - Execute the script to load, chunkify, and process the PDF document.

3. **Perform similarity search:**
    - Use the Faiss vector store to perform similarity searches and retrieve relevant documents based on your queries.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests to improve the project.

## License

This project is licensed under the GNU.
