from langchain_text_splitters import RecursiveCharacterTextSplitter

# Sample input text
text = """
Retrieval-Augmented Generation (RAG) combines information retrieval
with text generation. Chunking is a critical preprocessing step
that determines how documents are split before embedding.

Poor chunking leads to loss of context, while good chunking
improves retrieval accuracy and LLM response quality.
"""

# Initialize Recursive Character Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=["\n\n", "\n", " ", ""]
)

# Split text into chunks
chunks = text_splitter.split_text(text)

# Output chunks
for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}:\n{chunk}\n{'-'*50}")
