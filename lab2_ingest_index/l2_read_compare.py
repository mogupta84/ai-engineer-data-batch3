"""Day 2, code part 1: what a free loader reads from the 7 files, next to what Azure read into your own index.

    python lab2_ingest_index/l2_read_compare.py

Two free loaders on your own machine, LangChain's and LlamaIndex's, and on the right the text Foundry stored in your index,
<your account>-docs-source-index, which Document Intelligence read in the portal part (Standard mode).
Needs your Day 2 portal index. No model calls, no cost.
"""
import pathlib        # find the files
import sys
import warnings       # LangChain prints a notice about its newer packages; it does not change the result

warnings.filterwarnings("ignore")
from urllib.parse import unquote   # blob addresses write spaces and symbols as %xx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from azure_common import RAW, own_index, search_client


def free_loader(path):
    """LangChain's loader for the file's format; None when LangChain needs a much larger package for it."""
    from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
    if path.suffix == ".pdf":
        return "\n".join(d.page_content for d in PyPDFLoader(str(path)).load())
    if path.suffix == ".docx":
        return "\n".join(d.page_content for d in Docx2txtLoader(str(path)).load())
    return None                           # .pptx and .xlsx need the 'unstructured' package, about 1 GB


def llamaindex_read(path):
    """LlamaIndex's SimpleDirectoryReader picks a reader by file type. None when that reader needs more packages."""
    from llama_index.core import SimpleDirectoryReader
    try:
        return "\n".join(d.text for d in SimpleDirectoryReader(input_files=[str(path)]).load_data())
    except ImportError:
        return None


def azure_read():
    """Characters per file in your own index: Foundry may store a long file as several pieces, so they are added up."""
    totals = {}
    for p in search_client(own_index()).search(search_text="*", top=1000, select=["blob_url", "snippet"]):
        name = unquote(p["blob_url"].rsplit("/", 1)[-1])
        totals[name] = totals.get(name, 0) + len(p["snippet"] or "")
    return totals


def size(text):
    return "needs more" if text is None else f"{len(text.strip())}"


azure = azure_read()
print(f"{'File':38} {'LangChain':>10} {'LlamaIndex':>11} {'Your index':>11}")
for path in sorted(RAW.iterdir()):
    print(f"{path.name:38} {size(free_loader(path)):>10} {size(llamaindex_read(path)):>11} {azure.get(path.name, 0):>11}")
print("\nA 0 in a free column with no error is a silent failure: the scanned PDF has no text layer, only a picture.")
print("'needs more' means the library has a reader for that type, but only with further packages (about 1 GB for LangChain).")
