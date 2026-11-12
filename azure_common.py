"""Shared by the Day 1 to 4 labs: the settings, the Azure sign-in, the model clients, the FreshCart files, the
judged questions and the scoring. Every lab script imports it; you do not run it yourself.

The settings live in a file named .env in the repository folder (copy .env.example to .env and fill it in).
"""
import json           # read the judged questions and the saved pieces
import math           # the log in nDCG
import os             # read the settings
import pathlib        # find the course files
import re             # turn your name into a safe index name
import sys            # stop with a readable message when a setting is missing

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"                 # the 7 FreshCart files
GOLDEN = DATA / "golden"           # the judged questions
OUT = ROOT / "out"                 # what the labs write: read text, pieces, answers. Made on your machine only.

# What each file is: the labels every piece of text carries. The 2022 policy is the old one on purpose.
LABELS = {
    "refund_policy_v3_2024.pdf": {"doc_type": "policy", "version": "3.0", "effective_date": "2024-04-01", "current": True},
    "refund_policy_v2_2022.pdf": {"doc_type": "policy", "version": "2.0", "effective_date": "2022-06-15", "current": False},
    "partner_agreement_extract.pdf": {"doc_type": "contract", "version": "", "effective_date": "2023-01-01", "current": True},
    "support_sla_matrix.docx": {"doc_type": "sla", "version": "", "effective_date": "2024-04-01", "current": True},
    "support_escalation_sop_scanned.pdf": {"doc_type": "sop", "version": "5", "effective_date": "2024-03-01", "current": True},
    "support_qbr_q1_2024.pptx": {"doc_type": "report", "version": "", "effective_date": "2024-04-12", "current": True},
    "refund_claims_march_2024.xlsx": {"doc_type": "register", "version": "", "effective_date": "2024-04-01", "current": True},
}

# ---------------------------------------------------------------------------------------------------------------
# Settings: one NAME=value per line in .env. A value already set in the terminal wins.
# ---------------------------------------------------------------------------------------------------------------
ENV_FILE = ROOT / ".env"
if not ENV_FILE.exists() and ENV_FILE.with_name(".env.txt").exists():
    sys.exit("Your settings file is called .env.txt; rename it to .env (Windows added .txt when saving it).")
if ENV_FILE.exists():
    for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        name, _, value = line.partition("=")
        if name.strip() and not name.strip().startswith("#") and value.strip():
            os.environ.setdefault(name.strip(), value.strip().strip('"'))

DEFAULTS = {
    "FOUNDRY_OPENAI_ENDPOINT": "https://itc-ai-trainer-0359-resource.services.ai.azure.com/openai/v1",
    "CHAT_MODEL": "gpt-4.1-mini",
    "EMBED_MODEL": "text-embedding-3-small",
}


def setting(name, required=True):
    """One setting from .env, a default, or a clear message saying how to set it."""
    value = os.environ.get(name) or DEFAULTS.get(name)
    if not value and required:
        sys.exit(f"Add {name}=<the value from your trainer> to the .env file in the repository folder. "
                 f"See .env.example.")
    return value


def student():
    """Your account name with dashes, as in the portal steps: itc_ai_1 becomes itc-ai-1. It names your own index."""
    name = re.sub(r"[^a-z0-9-]", "", setting("STUDENT").lower().replace("_", "-"))[:30].strip("-")
    if not name:
        sys.exit("STUDENT in .env must be your account name, for example STUDENT=itc-ai-1")
    return name


# ---------------------------------------------------------------------------------------------------------------
# Sign-in: as you, with no key, through the Azure CLI: run az login once in the terminal you use.
# ---------------------------------------------------------------------------------------------------------------
_credential = None


def sign_in():
    global _credential
    if _credential is None:
        from azure.identity import AzureCliCredential
        _credential = AzureCliCredential()
    return _credential


def models():
    """The OpenAI client for the class's Foundry resource: chat models and the embedding model, one address."""
    from azure.identity import get_bearer_token_provider
    from openai import OpenAI
    token = get_bearer_token_provider(sign_in(), "https://cognitiveservices.azure.com/.default")
    return OpenAI(base_url=setting("FOUNDRY_OPENAI_ENDPOINT"), api_key=token)


def chat(client, messages, temperature=0.0, max_tokens=400, model=None):
    """One answer from the chat model, with the tokens it used. Returns (text, usage)."""
    r = client.chat.completions.create(model=model or setting("CHAT_MODEL"), messages=messages,
                                       temperature=temperature, max_tokens=max_tokens)
    return r.choices[0].message.content.strip(), r.usage


def embed(client, texts):
    """Each text as 1,536 numbers from the embedding model. Sends at most 100 texts a call."""
    out = []
    for i in range(0, len(texts), 100):
        out += [d.embedding for d in client.embeddings.create(model=setting("EMBED_MODEL"), input=texts[i:i + 100]).data]
    return out


def cosine(a, b):
    """How alike two lists of numbers point, from -1 to 1. Embedding models give values near 1 for similar text."""
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# ---------------------------------------------------------------------------------------------------------------
# Azure AI Search. A query key from your trainer if they gave one (it can only read), else your sign-in.
# ---------------------------------------------------------------------------------------------------------------
SHARED_INDEX = "freshcart-chunks"    # the trainer's index: 500-character pieces with labels, the same on both services


def own_index():
    """The index Foundry built from your container in the Day 2 portal part."""
    return f"{student()}-docs-source-index"


def _search_credential():
    key = os.environ.get("SEARCH_QUERY_KEY")
    if key:
        from azure.core.credentials import AzureKeyCredential
        return AzureKeyCredential(key)
    return sign_in()


def search_client(name=SHARED_INDEX):
    from azure.search.documents import SearchClient
    return SearchClient(setting("SEARCH_ENDPOINT"), name, _search_credential())


def search_admin():
    from azure.search.documents.indexes import SearchIndexClient
    return SearchIndexClient(setting("SEARCH_ENDPOINT"), _search_credential())


# ---------------------------------------------------------------------------------------------------------------
# Files the labs pass between days
# ---------------------------------------------------------------------------------------------------------------
def save_json(name, value):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=1, ensure_ascii=False), encoding="utf-8")


def load_json(name, made_by):
    path = OUT / name
    if not path.exists():
        sys.exit(f"{path} is missing. Run {made_by} first.")
    return json.loads(path.read_text(encoding="utf-8"))


def golden(name):
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


def question_vectors(client, questions):
    """The judged questions as numbers, saved after the first run so later runs do not pay for them again."""
    path = OUT / "question_vectors.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    missing = [q for q in questions if q not in saved]
    if missing:
        saved.update(zip(missing, embed(client, missing)))
        save_json("question_vectors.json", saved)
    return [saved[q] for q in questions]


# ---------------------------------------------------------------------------------------------------------------
# Scoring, as in the course's retrieval labs
# ---------------------------------------------------------------------------------------------------------------
def grade(rules, piece):
    """The judged grade of one piece: the best rule whose file matches and whose text the piece contains."""
    text = " ".join(piece["content"].lower().split())          # line breaks inside a sentence do not count
    return max([r["grade"] for r in rules
                if r["source"] == piece["file"] and " ".join(r["contains"].lower().split()) in text] or [0])


def ndcg_at_10(rules, ranked, every_piece):
    """How good the top 10 are, from 0 to 1: 1 means the best pieces in the index came first."""
    gains = [grade(rules, p) for p in ranked[:10]]
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    best = sorted([grade(rules, p) for p in every_piece], reverse=True)[:10]
    ideal = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(best))
    return dcg / ideal if ideal else None


def recall_at_10(rules, ranked, every_piece):
    """Of the pieces judged useful (grade 1 or more), the share that made the top 10."""
    useful = {p["id"] for p in every_piece if grade(rules, p) > 0}
    if not useful:
        return None
    return len(useful & {p["id"] for p in ranked[:10]}) / len(useful)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")
