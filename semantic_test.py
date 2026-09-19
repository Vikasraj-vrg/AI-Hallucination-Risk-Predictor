from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load AI language model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Two sentences
sentence1 = "Bell invented the telephone."
sentence2 = "Alexander Graham Bell is credited with inventing the telephone."

# Convert sentences into embeddings
embedding1 = model.encode([sentence1])
embedding2 = model.encode([sentence2])

# Calculate semantic similarity
similarity = cosine_similarity(embedding1, embedding2)[0][0]

print("Sentence 1:", sentence1)
print("Sentence 2:", sentence2)
print("Semantic Similarity:", round(similarity * 100, 2), "%")