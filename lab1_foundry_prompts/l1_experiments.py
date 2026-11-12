"""Day 1, code part 2: the playground experiments again, repeated and counted for you.

    python lab1_foundry_prompts/l1_experiments.py              all four experiments, about 45 model calls
    python lab1_foundry_prompts/l1_experiments.py temperature  just one: temperature, grounding, labels or embeddings

In the playground you ran each prompt once or twice and judged by eye. Here each runs several times, so you see how
often a result happens, not just whether it can. Signed in as you (az login), no key. Costs well under one cent.
"""
import json           # read the tickets
import pathlib        # find the files next to this script
import sys            # read which experiment to run

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))     # the repository folder, for azure_common
from azure_common import chat, cosine, embed, models, setting

HERE = pathlib.Path(__file__).resolve().parent
TEXTS = (HERE / "paste_texts.txt").read_text(encoding="utf-8")
TICKETS = json.loads((HERE / "tickets.json").read_text(encoding="utf-8"))


def block(heading):
    """One block of paste_texts.txt, by the start of its heading, so the code and the playground use the same text."""
    part = TEXTS.split(f"==== {heading}", 1)[1]
    return part.split("\n", 1)[1].split("\n====", 1)[0].strip()


def temperature(client):
    print("\n1. Temperature: the same question 5 times at each setting. How many different answers?")
    question = block("P2.")
    for t in (0.0, 1.0):
        answers = [chat(client, [{"role": "user", "content": question}], temperature=t, max_tokens=120)[0]
                   for _ in range(5)]
        print(f"   temperature {t}: {len(set(answers))} different answers out of 5. First: {answers[0][:100]!r}")
    print("   Temperature 0 is the most predictable setting, not a promise of the same words every time.")


def grounding(client):
    print("\n2. Grounding: the same questions with and without the policy, 3 times each.")
    policy = block("P3b.")
    strict = policy + "\n" + block("P3d.")
    runs = [("no documents", None, block("P3a.")),
            ("with the policy", policy, block("P3a.")),
            ("Singapore, with the policy", policy, block("P3c.")),
            ("Singapore, policy + 'I do not know' rule", strict, block("P3c."))]
    for name, system, question in runs:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
        answers = [chat(client, messages, max_tokens=120)[0] for _ in range(3)]
        three = sum("three" in a.lower() or " 3 " in f" {a} " for a in answers)
        unknown = sum("do not know" in a.lower() or "does not" in a.lower() for a in answers)
        print(f"   {name:42} says three: {three} of 3, says it does not know: {unknown} of 3")
        print(f"   {'':42} {answers[0][:110]!r}")


def labels(client):
    print("\n3. Prompt engineering: label 10 tickets, zero-shot then few-shot, 3 runs each.")
    user = block("P4b.")
    right = {t["id"]: t["label"] for t in TICKETS["tickets"]}
    for name, system in (("zero-shot", block("P4a.")), ("few-shot", block("P4a.") + "\n" + block("P4c."))):
        scores = []
        for _ in range(3):
            reply = chat(client, [{"role": "system", "content": system}, {"role": "user", "content": user}])[0]
            got = dict(line.replace(" ", "").split(":", 1) for line in reply.splitlines() if ":" in line)
            scores.append(sum(got.get(i) == label for i, label in right.items()))
        wrong = [f"{i} as {got.get(i)}" for i, label in right.items() if got.get(i) != label]
        print(f"   {name:10} right labels per run: {scores} out of 10. Wrong on the last run: {', '.join(wrong) or 'none'}")


def embeddings(client):
    print("\n4. Embeddings: how alike four sentences are, by meaning, from -1 to 1.")
    sentences = ["Can I get my money back for a spilled order?",
                 "Refund for leaked food delivery",
                 "How many refund claims can a customer make in thirty days?",
                 "The rider was very polite."]
    vectors = embed(client, sentences)
    print(f"   Each sentence becomes {len(vectors[0])} numbers.")
    for i in range(1, 4):
        print(f"   {cosine(vectors[0], vectors[i]):.3f}  {sentences[0]!r} vs {sentences[i]!r}")
    print("   Two sentences with no word in common can score higher than two that share the word 'refund'.")


EXPERIMENTS = {"temperature": temperature, "grounding": grounding, "labels": labels, "embeddings": embeddings}

if __name__ == "__main__":
    client = models()
    print(f"Model: {setting('CHAT_MODEL')}, embedding model: {setting('EMBED_MODEL')}")
    for name in (sys.argv[1:] or EXPERIMENTS):
        if name not in EXPERIMENTS:
            sys.exit(f"Unknown experiment {name!r}. Pick one of: {', '.join(EXPERIMENTS)}")
        EXPERIMENTS[name](client)
