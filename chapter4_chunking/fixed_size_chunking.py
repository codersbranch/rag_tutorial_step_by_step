def fixed_size_chunking(text, chunk_size=100):
    """
    Splits text into fixed-size chunks based on characters.
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


# Example text
text = (
    "Transformers are deep learning models that use attention mechanisms. "
    "They process all tokens in parallel, making them faster than RNNs. "
    "Transformers are widely used in NLP tasks."
)

chunks = fixed_size_chunking(text, chunk_size=80)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}:\n{chunk}\n")
