# Day 2: read, index and benchmark your documents

Start with the handbook, **[Day2_read_index_benchmark.pdf](Day2_read_index_benchmark.pdf)**. Parts P1 to P4 run in
Document Intelligence Studio, the Azure portal and Foundry; they build your own index. The code part reads that index.

```
Day2_read_index_benchmark.pdf  the handbook
l2_read_compare.py             a free loader against what Azure read into your index; no cost
l2_chunks.py                   three ways to cut the same text; no cost
l2_benchmark.py                Azure AI Search against pgvector on the same pieces: recall, nDCG, milliseconds
l2_scale.py                    the same two stores at 100,000 vectors: the ef dial, exact search, and a filter
scale_set.py                   makes the 100,000 vectors, the questions and the answer key from a fixed seed
trainer_build_shared_index.py  TRAINER ONLY, before class: builds the shared index freshcart-chunks
trainer_build_scale.py         TRAINER ONLY, before class: loads the 100,000 vectors into both stores
```

The 7 FreshCart files are in `data/raw/` at the top of the repository.
