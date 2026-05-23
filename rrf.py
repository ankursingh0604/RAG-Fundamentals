def rrf_fusion(bm25_results, semantic_results, k=60):
    rrf_scores = {}

    # 1. Score documents from the BM25 list
    for rank, doc in enumerate(bm25_results, start=1):
        if doc not in rrf_scores:
            rrf_scores[doc] = 0.0
        rrf_scores[doc] += 1 / (rank + k)

    # 2. Score documents from the Semantic list
    for rank, doc in enumerate(semantic_results, start=1):
        if doc not in rrf_scores:
            rrf_scores[doc] = 0.0
        rrf_scores[doc] += 1 / (rank + k)

    # 3. Sort the documents from highest RRF score to lowest
    final_ranking = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    return final_ranking

bm25_list = ["Document A", "Document B", "Document C"]
semantic_list = ["Document C", "Document A", "Document D"]

final_fused_results = rrf_fusion(bm25_list, semantic_list, k=60)

for doc, score in final_fused_results:
    print(f"{doc}: Score = {score:.5f}")