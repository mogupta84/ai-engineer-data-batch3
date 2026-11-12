"""Day 1, code part 1: how a model counts text. No Azure call, no cost.

    python lab1_foundry_prompts/l1_tokens.py

A model reads tokens, not words: pieces of words from a fixed list. You pay per token, and the context window is
measured in tokens. gpt-4.1-mini uses the list called o200k_base.
"""
import tiktoken       # the tokenizer OpenAI's models use

PRICE_IN, PRICE_OUT = 0.40, 1.60       # US dollars per million tokens for gpt-4.1-mini (Global Standard, list price)

SENTENCES = {
    "English": "A customer may make up to three refund claims in any rolling thirty day period.",
    "Hindi": "एक ग्राहक किसी भी तीस दिन की अवधि में अधिकतम तीन रिफंड दावे कर सकता है।",
    "Ticket id": "FCL-2024-03-118842 UPI ref 4027 1188 4231",
}

enc = tiktoken.get_encoding("o200k_base")
for name, text in SENTENCES.items():
    tokens = enc.encode(text)
    pieces = [enc.decode([t]) for t in tokens]
    print(f"{name:10} {len(text.split()):3} words  {len(text):3} characters  {len(tokens):3} tokens")
    print("           " + " | ".join(pieces[:14]) + (" | ..." if len(pieces) > 14 else ""))

# What a typical support answer costs: a 1,700-character policy plus a question in, a 60-word answer out.
policy_tokens, answer_tokens = 400, 90
one = policy_tokens * PRICE_IN / 1e6 + answer_tokens * PRICE_OUT / 1e6
print(f"\nOne grounded answer: about {policy_tokens} tokens in and {answer_tokens} out, ${one:.5f}.")
print(f"10,000 answers a day for 30 days: ${one * 300_000:,.0f} a month.")
