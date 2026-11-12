"""Day 2, code part 2: three ways to cut the same text into pieces, and what each does to one answer.

    python lab2_ingest_index/l2_chunks.py

The portal's Standard mode stored a few large pieces per file. Search returns pieces, and a model reads what search
returns, so the cut decides what any later step can see. This uses the text in your own index; no cost.
"""
import pathlib
import sys
from urllib.parse import unquote

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from azure_common import own_index, search_client
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

text = {}
for p in search_client(own_index()).search(search_text="*", top=1000, select=["blob_url", "snippet"]):
    name = unquote(p["blob_url"].rsplit("/", 1)[-1])
    text[name] = text.get(name, "") + "\n" + (p["snippet"] or "")

by_heading = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False)
WAYS = {
    "whole file, as Azure read it": lambda t: [t],
    "1,500 characters, 150 overlap": RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150).split_text,
    "500 characters, 50 overlap": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text,
    "by heading": lambda t: [d.page_content for d in by_heading.split_text(t)],
}
FACT = "up to three refund claims"           # the answer to the most asked question, in the 2024 policy

print(f"{'Way to cut':32} {'pieces':>6} {'average size':>13}   the piece holding '{FACT}'")
for name, cut in WAYS.items():
    pieces = [(f, piece) for f, t in text.items() for piece in cut(t) if piece.strip()]
    sizes = [len(piece) for _, piece in pieces]
    holder = next((piece for f, piece in pieces if f.startswith("refund_policy_v3") and FACT in " ".join(piece.split())), "")
    print(f"{name:32} {len(pieces):>6} {sum(sizes) / len(sizes):>13.0f}   {len(holder):>5} characters, "
          f"of which the fact is {len(FACT) / len(holder):.0%}" if holder else f"{name:32} {len(pieces):>6}   not found")
print("\nSmaller pieces: the fact is a bigger share of what search returns, so meaning search matches it better,")
print("but a piece can lose the heading that says which policy it is from. The trainer's shared index uses 500.")
