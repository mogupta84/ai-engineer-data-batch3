# Day 2: load, index and benchmark your documents

Start with the handbook, **[Day2_read_index_benchmark.pdf](Day2_read_index_benchmark.pdf)**. The day runs in this
order: notebook 1 in Microsoft Fabric, then parts P1 to P4 in the Azure portal, then notebook 2 in Fabric.

```
Day2_read_index_benchmark.pdf  the handbook
day2_nb1_ingestion.ipynb       N1: load the 7 files yourself and find what silently goes wrong (import into Fabric)
day2_nb2_benchmark.ipynb       N2: Azure AI Search against PostgreSQL with pgvector, measured inside Azure (import into Fabric)
scale_set.py                   makes the benchmark vectors, questions and answer key from a fixed seed (notebook 2 fetches it)
trainer_build_scale.py         TRAINER ONLY, before class: loads the benchmark vectors into both stores (add --1536 for the second set)
trainer_build_shared_index.py  TRAINER ONLY, before class: builds the shared index freshcart-chunks that Days 3 and 4 read

```

The 7 FreshCart files are in `data/raw/`. The notebooks fetch the lab code in `src/` from this repository.
