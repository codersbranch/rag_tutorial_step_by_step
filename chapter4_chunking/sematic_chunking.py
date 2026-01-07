import numpy as np
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_chunking(text, similarity_threshold=0.65):
    sentences = sent_tokenize(text)
    embeddings = model.encode(sentences, normalize_embeddings=True)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i], embeddings[i - 1])

        if similarity >= similarity_threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    chunks.append(" ".join(current_chunk))
    return chunks


# Example text
text = """
Transformers use attention mechanisms to process language.
They handle long-range dependencies efficiently.
They are widely used in NLP tasks.

RNNs process text sequentially.
They struggle with long contexts.
"""

chunks = semantic_chunking(text)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}:\n{chunk}\n")
