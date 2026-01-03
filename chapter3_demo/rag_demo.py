# Step 1: Install required libraries
# pip install sentence-transformers faiss-cpu numpy

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Step 2: Sample documents
documents = [
    "Transformers use self-attention mechanism",
    "RAG combines retrieval and generation",
    "FAISS is used for vector similarity search",
    "Embeddings represent text as vectors"
]


# Step 3: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Generate embeddings
embeddings = model.encode(documents)
print("Original embeddings shape:", embeddings.shape)

# Step 5: Normalize embeddings (important for cosine similarity)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Step 6: Create FAISS index (cosine similarity via inner product)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

# Step 7: Add embeddings to index
index.add(embeddings)
print("Number of vectors in index:", index.ntotal)

# Step 8: Query example
query = "What is RAG?"
query_embedding = model.encode([query])
query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

# Step 9: Search
top_k = 2
scores, indices = index.search(query_embedding, top_k)

print("\nQuery:", query)
print("Top matches:")
for i in indices[0]:
    print("-", documents[i])
