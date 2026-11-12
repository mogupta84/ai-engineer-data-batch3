# AI Engineer (Data), Batch 3: the labs

The labs of the AI Engineer (Data) course, 12 days over 6 weeks, all on Microsoft Azure: Microsoft Foundry for models
and agents, Azure AI Search for documents, Document Intelligence for reading files, and Microsoft Fabric for data
pipelines. Each day has a folder and a handbook (PDF) with every step and the numbers to expect: open the handbook first.

Every lab starts in the browser, in the Azure and Foundry portals. Where the portal cannot do the job (counting results
over many runs, scoring search quality, running an API), the lab moves to a short code part in VS Code.

## Day by day

| Day | Lab | What you do | Folder and handbook |
|---|---|---|---|
| 1 | 1 | Meet gpt-4.1-mini in Foundry: a deployment, temperature, a made-up answer and the same question grounded, zero-shot and few-shot prompts; then count the same experiments in code | [`lab1_foundry_prompts`](lab1_foundry_prompts), [Day1_models_and_prompts.pdf](lab1_foundry_prompts/Day1_models_and_prompts.pdf) |
| 2 | 2 | Read 7 files in 5 formats with Document Intelligence, index them in your own Azure AI Search index from Foundry, count what went in; compare loaders and ways to cut, and benchmark Azure AI Search against pgvector | [`lab2_ingest_index`](lab2_ingest_index), [Day2_read_index_benchmark.pdf](lab2_ingest_index/Day2_read_index_benchmark.pdf) |
| 3 | 3 | Hybrid search, the semantic reranker and a metadata filter in Search explorer, a RAG agent in Foundry; score five stages of search with nDCG@10 and read a RAG pipeline end to end | [`lab3_rag_rerank`](lab3_rag_rerank), [Day3_rag_hybrid_rerank.pdf](lab3_rag_rerank/Day3_rag_hybrid_rerank.pdf) |
| 4 | 4 | Evaluate your agent in Foundry, change it and compare runs; ship the RAG pipeline as an API whose citations are checked in code, and measure a fix | [`lab4_rag_api`](lab4_rag_api), [Day4_rag_api_and_evaluation.pdf](lab4_rag_api/Day4_rag_api_and_evaluation.pdf) |
| 5 | 5 | Turn your own Python function into a tool an agent can call, first as an OpenAPI tool, then as an MCP server | [`lab5_foundry`](lab5_foundry), [Foundry_grounding_and_MCP_demo.pdf](lab5_foundry/Foundry_grounding_and_MCP_demo.pdf) |
| 6 | 6 | Ask questions in English, let a model write the SQL, and make its numbers ones you can defend with business rules and a semantic layer | [`lab6_foundry`](lab6_foundry), [Foundry_Lab6_numbers_you_can_defend.pdf](lab6_foundry/Foundry_Lab6_numbers_you_can_defend.pdf) |
| 7 | 7 | Build a first enterprise agent in the Foundry portal, with tools, a plan and memory, plus a look at skills | [`lab7_foundry`](lab7_foundry), [Foundry_Lab7_first_enterprise_agent.pdf](lab7_foundry/Foundry_Lab7_first_enterprise_agent.pdf) |
| 8 | 8 | Build a support agent in code with Microsoft Agent Framework, add tracing, memory, a tool and a skill, then a router, and score it after each step | [`lab8_foundry`](lab8_foundry), [Foundry_Lab8_agent_in_code.pdf](lab8_foundry/Foundry_Lab8_agent_in_code.pdf) |
| 9 | 9 | Build a ticket pipeline in Microsoft Fabric: copy only new rows, park bad ones with a reason, label each ticket with a model, and ask the tables questions with a data agent | [`lab9_fabric`](lab9_fabric), [Fabric_Lab9_pipeline.pdf](lab9_fabric/Fabric_Lab9_pipeline.pdf) |
| 10 | 10 | Add alarms to your agent and your pipeline: a quality check, traces, health checks with an owner each, embedding drift and MLflow, each tested by breaking something on purpose | [`lab10_monitoring`](lab10_monitoring), [Lab10_knowing_when_it_breaks.pdf](lab10_monitoring/Lab10_knowing_when_it_breaks.pdf) |
| 11 | 11 | Take personal data out before it is stored, give each team access to only its own documents, and keep an audit record of who saw what | [`lab11_governance`](lab11_governance), [Lab11_who_may_see_what.pdf](lab11_governance/Lab11_who_may_see_what.pdf) |
| 12 | Capstone | Build and present an enterprise data agent on Foundry: ingestion, AI Search, RAG, tools, NL2SQL, monitoring, evaluation and access control. Your trainer gives you the capstone brief | |

## Before Day 1

- **Your class account.** You sign in to ai.azure.com and portal.azure.com with the account your trainer gives you.
  The whole class shares one Foundry project, so name everything you create after your account and open only your own.
- **For the code parts** you need Python 3.11 or newer, VS Code and the Azure CLI on your laptop. The Day 1 handbook
  sets them up once for Days 1 to 4, in about 20 minutes. On Windows, `setup\setup_windows.cmd` does the Python part
  in one go. Labs 5 to 11 have their own setup in their handbooks.
- **Sign-in, no keys.** The code signs in as you with `az login`. If it fails with a device management message, run
  `az config set core.enable_broker_on_windows=false` and `az login` again. Settings your trainer gives you go in a
  file named `.env` (copy `.env.example`); it never goes to GitHub.

## What is in this repository

```
lab1_foundry_prompts/  Day 1: models, temperature, grounding, prompts; tokens and embeddings in code
lab2_ingest_index/     Day 2: Document Intelligence, Foundry knowledge, your index; loaders, cutting, pgvector benchmark
lab3_rag_rerank/       Day 3: hybrid search, reranker, filters, a RAG agent; nDCG@10 per stage, the RAG pipeline
lab4_rag_api/          Day 4: Foundry evaluation of your agent; the RAG API and its scoring
lab5_foundry/ to lab11_governance/   Days 5 to 11, each with its handbook
azure_common.py        the settings, sign-in, model calls and scoring shared by the Day 1 to 4 code
data/raw/              the 7 FreshCart documents
data/golden/           the judged questions for Days 2 to 4
data/freshcart.db      the FreshCart orders warehouse, one SQLite file (Lab 7 loads it into Azure SQL)
setup/                 the one-command Windows setup for Days 1 to 4
requirements.txt       the Python packages for Days 1 to 4
.env.example           the settings template for Days 1 to 4
```

The data is a fictional food delivery company, FreshCart: two refund policies (one withdrawn), a partner contract,
a scanned procedure, a Word table, a slide deck, a spreadsheet and an orders warehouse.

Course material for participants of the AI Engineer (Data) programme.
