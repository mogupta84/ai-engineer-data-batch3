"""Azure for the Day 2 Fabric notebooks: sign in as the student, read files with Document Intelligence, and embed
with the class embedding model.

In a Microsoft Fabric notebook the sign-in is the notebook's own (notebookutils), so there is no key. On a laptop it
falls back to the Azure CLI (az login), so the same notebook also runs in VS Code.
"""
from __future__ import annotations

import logging
import pathlib
import time

import numpy as np

FOUNDRY = "https://itc-ai-trainer-0359-resource.services.ai.azure.com"    # the class Foundry resource
DOCINTEL = "https://freshcart-docintel-itc.cognitiveservices.azure.com/"    # the class Document Intelligence
EMBED_MODEL = "text-embedding-3-small"
SCOPE = "https://cognitiveservices.azure.com"
for _name in ("azure", "httpx", "openai"):                  # keep each request's log lines off the screen
    logging.getLogger(_name).setLevel(logging.WARNING)


def _fabric():
    """Fabric's notebook tools, or None on a laptop. Imported, because a module cannot see the notebook's own names."""
    try:
        import notebookutils
        return notebookutils
    except ImportError:
        return None


class SignIn:
    """A credential the Azure SDKs accept, built on the Fabric notebook's token for Azure AI services."""

    def get_token(self, *scopes, **kwargs):
        from azure.core.credentials import AccessToken
        tools = _fabric()
        if tools is None:                                         # a laptop: az login
            from azure.identity import AzureCliCredential
            return AzureCliCredential().get_token(SCOPE + "/.default")
        return AccessToken(tools.credentials.getToken(SCOPE), int(time.time()) + 600)


def in_fabric() -> bool:
    return _fabric() is not None


def document_intelligence(paths: list[pathlib.Path]) -> tuple[dict, float]:
    """Each file through Document Intelligence's layout model, as text with headings and tables kept.
    Returns ({file: {"text", "pages", "tables"}}, seconds). The layout model is paid per page: the 7 files are 9
    pages, under $0.10 at list price."""
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import DocumentContentFormat
    client = DocumentIntelligenceClient(DOCINTEL, SignIn())
    out, start = {}, time.perf_counter()
    for path in paths:
        result = client.begin_analyze_document("prebuilt-layout", body=path.read_bytes(),
                                               output_content_format=DocumentContentFormat.MARKDOWN).result()
        out[path.name] = {"text": result.content or "", "pages": len(result.pages or []),
                          "tables": len(result.tables or [])}
    return out, time.perf_counter() - start


class AzureEmbedder:
    """The class embedding model, text-embedding-3-small: 1,536 numbers per piece. Same shape as the other embedders
    in freshcart.py (fit, encode, name), so VectorIndex takes it unchanged. It needs no fitting: a piece's vector
    depends only on its own text."""
    name = EMBED_MODEL

    def __init__(self):
        from openai import OpenAI
        sign_in = SignIn()
        self.client = OpenAI(base_url=f"{FOUNDRY}/openai/v1",
                             api_key=lambda: sign_in.get_token(SCOPE + "/.default").token)
        self._cache: dict[str, list[float]] = {}

    def fit(self, texts):
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for i in range(0, len(todo), 64):
            batch = todo[i:i + 64]
            for text, d in zip(batch, self.client.embeddings.create(model=EMBED_MODEL, input=batch).data):
                self._cache[text] = d.embedding
        V = np.array([self._cache[t] for t in texts], dtype="float32")
        return V / np.linalg.norm(V, axis=1, keepdims=True)
