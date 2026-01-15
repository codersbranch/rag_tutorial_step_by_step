from langchain_text_splitters import CharacterTextSplitter

text = '''The United States offers a diverse range of tourist destinations that cater to every type of traveler. 
Cities such as New York and Los Angeles attract visitors with iconic landmarks,
cultural institutions, and vibrant entertainment scenes, while natural attractions like the Grand Canyon, Yellowstone National Park, and Yosemite showcase some of the world’s most dramatic landscapes. Coastal destinations, including Florida’s beaches and California’s Pacific coastline, are popular for relaxation and outdoor activities, whereas historic cities like Washington, D.C. and Boston provide insight into the nation’s history through monuments, museums, and preserved architecture. This variety of urban, natural, and historical attractions makes the USA one of the most versatile and appealing tourist destinations globally.'''

text_splitter = CharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separator="\n"
)

chunks = text_splitter.split_text(text)
print("Generated Chunks:",chunks)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")
