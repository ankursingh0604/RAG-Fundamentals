from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_retriever(query, corpus, top_n=2):
    # 1. Convert corpus and query to embeddings (vectors)
    corpus_embeddings = model.encode(corpus)
    query_embedding = model.encode([query])

    # 2. Calculate cosine similarity between query and each doc
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]

    # 3. Rank documents by similarity score
    ranked_indices = np.argsort(similarities)[::-1][:top_n]

    return [corpus[i] for i in ranked_indices]

# Test
query = "How tall is the Eiffel Tower?"
print(semantic_retriever(query, corpus))