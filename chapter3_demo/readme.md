
# RAG Embedding, Normalization & Indexing Demo

This project demonstrates the **core building blocks of Retrieval-Augmented Generation (RAG)**:
**Embedding**, **Normalization**, and **Vector Indexing** using Python.

The example uses a small set of sample documents and shows how similarity search works using **FAISS**.

---

## Project Overview

The pipeline implemented in this demo:

1. Convert text documents into embeddings using a Transformer model  
2. Normalize embeddings for cosine similarity  
3. Store embeddings in a FAISS index  
4. Search for the most relevant documents for a query  

This forms the foundation of **semantic search** and **RAG systems**.

---

## Technologies Used

- **sentence-transformers** – Generate semantic embeddings  
- **FAISS** – Fast vector similarity search  
- **NumPy** – Vector normalization and numerical operations  

---

## Project Structure

```
.
├── rag_demo.py
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

- **Windows**
```bash
venv\Scripts\activate
```

- **macOS / Linux**
```bash
source venv/bin/activate
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Demo

```bash
python rag_demo.py
```

You should see:
- Embedding shape
- Number of vectors indexed
- Top matching documents for the query

---

## Requirements

- Python 3.8+
- CPU-based FAISS (GPU optional)

---

## Notes

- This demo focuses on **retrieval only**
- LLM generation can be added on top of this pipeline
- Suitable for beginners learning RAG fundamentals

---

## License

This project is provided for educational purposes.
