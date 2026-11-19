# Day 3: RAG with hybrid search and a reranker

Start with the handbook, **[Day3_rag_hybrid_rerank.pdf](Day3_rag_hybrid_rerank.pdf)**. Parts P1 to P3 run in Search
explorer and Foundry, on your Day 2 index and knowledge base.

```
Day3_rag_hybrid_rerank.pdf  the handbook
paste_texts.txt             the JSON blocks P1a to P1d, the filter pattern and the agent texts for P1 to P3
l3_search.py                six stages of search scored with nDCG@10 and recall@10 on 20 questions: you fill 3 gaps
l3_search_solution.py       the same file with the gaps filled, if you are stuck
l3_rag.py                   the RAG pipeline step by step: you write the augment step (GAP 4)
l3_rag_solution.py          the same file with GAP 4 filled, if you are stuck
rag.py                      the pipeline itself: search, rerank, gate, answer, check citations (Day 4 uses it too)
```
