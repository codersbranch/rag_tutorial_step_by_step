import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Simple regex-based sentence splitter
def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

# Sample document
text = """
RAG combines retrieval and generation to improve LLM accuracy.
Chunking is a crucial preprocessing step in RAG systems.

Embeddings convert text into numerical vectors.
Cosine similarity measures how close two vectors are.

Vector databases store embeddings efficiently.
They enable fast semantic search at scale.
"""

# Step 1: Sentence splitting
sentences = split_sentences(text)

# Step 2: Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

# Step 3: Compare adjacent sentence similarity
similarities = cosine_similarity(embeddings[:-1], embeddings[1:])

# Step 4: Semantic chunking
chunks = []
current_chunk = [sentences[0]]
THRESHOLD = 0.6

for i in range(1, len(sentences)):
    if similarities[i - 1][0] >= THRESHOLD:
        current_chunk.append(sentences[i])
    else:
        chunks.append(" ".join(current_chunk))
        current_chunk = [sentences[i]]

chunks.append(" ".join(current_chunk))

# Output chunks
for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}:\n{chunk}\n{'-' * 50}")
