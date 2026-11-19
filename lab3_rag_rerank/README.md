# Day 3: RAG with hybrid search and a reranker

Start with the handbook, **[Day3_rag_hybrid_rerank.pdf](Day3_rag_hybrid_rerank.pdf)**. Parts P1 to P3 run in Search
explorer and Foundry, on your Day 2 index and knowledge base. The code part is trainer-led: your trainer runs the two
solution scripts on screen. You can run them too, without --rerank (the reranker's free searches are shared by the class).

```
Day3_rag_hybrid_rerank.pdf  the handbook
paste_texts.txt             the JSON blocks P1a to P1d, P2a and P2b, and the agent texts for P1 to P3
l3_search_solution.py       six stages of search scored with nDCG@10 and recall@10 on 20 questions (run in class)
l3_rag_solution.py          the RAG pipeline step by step: retrieve, augment, generate, check (run in class)
l3_search.py, l3_rag.py     the same two files with gaps to fill yourself, for practice after class
rag.py                      the pipeline itself: search, rerank, gate, answer, check citations (Day 4 uses it too)
```
