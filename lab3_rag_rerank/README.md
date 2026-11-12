# Day 3: RAG with hybrid search and a reranker

Start with the handbook, **[Day3_rag_hybrid_rerank.pdf](Day3_rag_hybrid_rerank.pdf)**. Parts P1 to P3 run in Search
explorer and Foundry, on your Day 2 index and knowledge base.

```
Day3_rag_hybrid_rerank.pdf  the handbook
paste_texts.txt             the question, the filter line and the agent instructions to paste in P1 to P3
l3_search.py                five stages of search, each scored with nDCG@10 on 20 judged questions
l3_rag.py                   one question through the RAG pipeline, with the pieces it read
rag.py                      the pipeline itself: search, rerank, gate, answer, check citations (Day 4 uses it too)
```
