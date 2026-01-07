import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

def recursive_chunk(text, max_length=150):
    """
    Recursively splits text into chunks smaller than max_length
    using paragraph → sentence → word → character fallback.
    """
    chunks = []

    # Step 1: Split by paragraphs
    paragraphs = text.split("\n\n")

    for para in paragraphs:
        if len(para) <= max_length:
            chunks.append(para)
        else:
            # Step 2: Split by sentences
            sentences = sent_tokenize(para)
            current_chunk = ""

            for sent in sentences:
                if len(current_chunk) + len(sent) <= max_length:
                    current_chunk += " " + sent
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent

            if current_chunk:
                chunks.append(current_chunk.strip())

    return chunks


# Example text
text = """
Transformers are deep learning models that use attention mechanisms.
They handle long-range dependencies better than RNNs.
They process tokens in parallel, which improves training speed.

Transformers are widely used in translation, summarization, and question answering.
"""

chunks = recursive_chunk(text, max_length=120)

for i, c in enumerate(chunks, 1):
    print(f"Chunk {i}:\n{c}\n")
