def structure_based_chunking(text):
    """
    Splits text using headings as chunk boundaries.
    Assumes headings start with '##'
    """
    chunks = []
    current_chunk = ""

    for line in text.split("\n"):
        if line.startswith("##"):
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# Example document
text = """
## Introduction
Transformers are neural network models used in NLP.

## Architecture
They use self-attention and feed-forward layers.

## Applications
Transformers are used in translation and summarization.
"""

chunks = structure_based_chunking(text)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}:\n{chunk}\n")
