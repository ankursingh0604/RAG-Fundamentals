from rank_bm25 import BM25Okapi

def bm25_retriever(query, corpus, top_n=2):
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    top_docs = bm25.get_top_n(tokenized_query, corpus, n=top_n)

    return top_docs


corpus = [
    "The capital of France is Paris.",
    "Quantum computing utilizes qubits.",
    "The Eiffel Tower is 330 meters tall."
]

query = "How tall is the Eiffel Tower?"
print(bm25_retriever(query, corpus))