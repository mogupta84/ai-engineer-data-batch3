# Day 4: a RAG API with verifiable citations, and Foundry evaluation

Start with the handbook, **[Day4_rag_api_and_evaluation.pdf](Day4_rag_api_and_evaluation.pdf)**. Parts P1 to P3 run
in Foundry on your Day 3 agent.

```
Day4_rag_api_and_evaluation.pdf  the handbook
day4_eval_cases.jsonl            P2: 10 test cases (query, ground_truth) to upload as the evaluation's dataset
l4_api.py                        the Day 3 pipeline as an API; open http://127.0.0.1:8000/docs; --my-check uses your check
l4_checks.py                     the citation check: you fill 3 gaps, then run it (9 tests, no Azure)
l4_checks_solution.py            the same file with the gaps filled, if you are stuck
my_prompt.txt                    your own prompt for --prompt mine (starts as prompt v1)
paste_texts.txt                  the trainer's P3 sentence and the API question, if you are stuck
l4_score.py                      25 judged questions; --fake, --prompt v2, --prompt mine, --evaluate, --upload
```
