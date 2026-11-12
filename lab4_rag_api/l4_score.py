"""Day 4, code part 2: score the RAG pipeline on 25 judged questions, then with Foundry's evaluators.

    python lab4_rag_api/l4_score.py                    20 answerable and 5 unanswerable questions, scored by rules
    python lab4_rag_api/l4_score.py --fake             add a made-up citation to every third answer: is it caught?
    python lab4_rag_api/l4_score.py --prompt v2        the stricter prompt (see PROMPT_V2 below)
    python lab4_rag_api/l4_score.py --evaluate         also Groundedness, Relevance and a right-answer judge (gpt-4.1-mini)
    python lab4_rag_api/l4_score.py --evaluate --upload  the same, and the run appears in Foundry, Evaluations

Rules check facts you can name ("three"). Evaluators judge what rules cannot: is every sentence supported by the
pieces (groundedness), does it answer the question (relevance). Both scored 1 to 5.
Cost: about 25 short answers, plus about 70 judge calls with --evaluate: well under $0.10.
The judge is gpt-4.1-mini, the same model that writes the answers: a model grading its own work.
"""
import contextlib     # hide the evaluators' progress lines
import io
import json
import logging
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab3_rag_rerank"))
from azure_common import OUT, chat, golden, models, setting, sign_in
from rag import PROMPT, REFUSAL, answer, verify

PROMPT_V2 = PROMPT + """
Before answering, check that the pieces cover the exact place, product and time the question asks about. FreshCart's
policies cover food delivery orders in India: if the question is about another country or another kind of order,
reply exactly with the sentence above, even when the pieces discuss refunds in general."""


def has_fact(text, expected):
    """Whether the answer states one of the expected facts, as a whole word or phrase. Citation ids are removed
    first: an id such as [3f37a93f2727] must not count as the number 3."""
    text = re.sub(r"\[[^\]]*\]", " ", text).lower()
    text = text.replace(chr(8211), "-").replace(chr(8212), "-")      # a range written with a long dash counts as 5-7
    def found(e):                     # short words and numbers match whole; a long stem such as "escalat" may go on
        end = "" if len(e) >= 6 and not e[-1].isdigit() else r"(?![\w]|\.\d)"   # 18 must not match 18.4
        return re.search(r"(?<![\w.])" + re.escape(e.lower()) + end, text)
    return any(found(e) for e in expected)


def option(name, default=None):
    """The value after --name, or default when it is missing."""
    i = sys.argv.index(name) if name in sys.argv else -1
    value = sys.argv[i + 1] if 0 <= i < len(sys.argv) - 1 else None
    return default if value is None or value.startswith("--") else value


def run(client, prompt):
    qa = golden("qa_golden.json")
    rows, fakes, caught = [], 0, 0
    for n, q in enumerate(qa["answerable"] + qa["unanswerable"]):
        answerable = "expect_any" in q
        r = answer(client, q["question"], prompt=prompt)
        right = answerable and not r["refused"] and has_fact(r["answer"], q["expect_any"])
        if "--fake" in sys.argv and answerable and n % 3 == 0 and not r["refused"]:
            # A copy with a made-up source added: the score above and the saved answer stay as the model wrote them.
            faked = verify({**r, "answer": r["answer"] + " [deadbeef0000]"})
            fakes += 1
            caught += "deadbeef0000" in faked["unknown_citations"]
            print(f"{q['id']} made-up source added: {'CAUGHT, ' + faked['refusal_reason'] if faked['refused'] else 'NOT CAUGHT'}")
        verdict = ("RIGHT" if right else "refused" if r["refused"] else "WRONG") if answerable else \
                  ("refused, as it should" if r["refused"] else "ANSWERED")
        print(f"{q['id']} {verdict:22} {r['refusal_reason'] or ''}")
        rows.append({**q, **r, "right": right, "answerable": answerable})
    a = [r for r in rows if r["answerable"]]
    u = [r for r in rows if not r["answerable"]]
    print(f"\nRight facts: {sum(r['right'] for r in a)} of {len(a)}. Refused though answerable: "
          f"{sum(r['refused'] for r in a)} of {len(a)}. Refused, as it should: {sum(r['refused'] for r in u)} of {len(u)}.")
    if fakes:
        print(f"Made-up citations added: {fakes}, caught by the code check: {caught}")
    return rows


JUDGE_PROMPT = """You check one answer from a support assistant against what a correct answer must do.
Question: {query}
What a correct answer must do: {expected}
The assistant's answer: {response}
Reply with PASS or FAIL on the first line, then one sentence saying why."""


class RightAnswer:
    """A custom evaluator, the same idea as Lab 10's right-decision judge: it compares each answer with what the answer
    key says it must do, which groundedness and relevance never look at. gpt-4.1-mini is the judge."""

    def __init__(self):
        self.client = models()

    def __call__(self, *, query, response, expected):
        verdict, _usage = chat(self.client, [{"role": "user", "content": JUDGE_PROMPT.format(
            query=query, expected=expected, response=response)}], max_tokens=80)
        word = re.search(r"\b(PASS|FAIL)\b", verdict.upper())          # tolerate **PASS** or "Verdict: FAIL"
        return {"right_answer": None if word is None else int(word.group(1) == "PASS"), "right_answer_reason": verdict}


def expected_for(row):
    """What the answer key says a correct answer must do, in words the judge can check."""
    if row["answerable"]:
        return "give this answer from FreshCart's documents, in any words: " + row["expected"]
    return ("say that FreshCart's documents do not answer this question. Giving a policy, a number or advice that "
            "the documents do not state for this exact case is a FAIL, even if it cites a document.")


def evaluate(rows, name):
    """Foundry's Groundedness and Relevance evaluators, and the right-answer judge, on every answer not refused."""
    from azure.ai.evaluation import AzureOpenAIModelConfiguration, GroundednessEvaluator, RelevanceEvaluator, evaluate
    data = OUT / f"{name}.jsonl"
    with data.open("w", encoding="utf-8") as f:
        for r in rows:
            if not r["refused"]:
                context = "\n\n".join(f"[{p['id']}] {p['text']}" for p in r["retrieved"])
                f.write(json.dumps({"query": r["question"], "response": r["answer"], "context": context,
                                    "expected": expected_for(r)}) + "\n")
    judge = AzureOpenAIModelConfiguration(azure_endpoint=setting("FOUNDRY_OPENAI_ENDPOINT").split("/openai")[0],
                                          azure_deployment=setting("CHAT_MODEL"), api_version="2024-10-21")
    upload = {"azure_ai_project": setting("FOUNDRY_PROJECT_ENDPOINT")} if "--upload" in sys.argv else {}
    print("\nJudging the answers (about a minute)...")
    logging.disable(logging.WARNING)                       # the evaluators log every line; keep the screen readable
    with contextlib.redirect_stdout(io.StringIO()):
        result = evaluate(data=str(data), evaluation_name=name, output_path=str(OUT / f"{name}_result.json"),
                          evaluators={"groundedness": GroundednessEvaluator(judge, credential=sign_in()),
                                      "relevance": RelevanceEvaluator(judge, credential=sign_in()),
                                      "right": RightAnswer()}, **upload)
    logging.disable(logging.NOTSET)
    m = result["metrics"]
    print(f"\nFoundry evaluators on {sum(1 for _ in data.open())} answers, 1 to 5: "
          f"groundedness {m.get('groundedness.groundedness', float('nan')):.2f}, "
          f"relevance {m.get('relevance.relevance', float('nan')):.2f}")
    passed = [row.get("outputs.right.right_answer") for row in result["rows"]]
    unclear = sum(p not in (0, 1) for p in passed)
    print(f"Right-answer judge: {sum(p == 1 for p in passed)} of {len(passed)} answers not refused pass"
          + (f", {unclear} unclear" if unclear else ""))
    for row in result["rows"]:
        if (row.get("outputs.groundedness.groundedness") or 5) < 4:
            print(f"  low groundedness: {row['inputs.query'][:70]!r}")
        if row.get("outputs.right.right_answer") == 0:
            reason = (row.get("outputs.right.right_answer_reason") or "").strip().splitlines() or [""]
            print(f"  judge FAIL: {row['inputs.query'][:60]!r}: {reason[-1][:110]}")
    if result.get("studio_url"):
        print(f"Open the run in Foundry: {result['studio_url']}")


if __name__ == "__main__":
    v2 = option("--prompt") == "v2"
    rows = run(models(), PROMPT_V2 if v2 else PROMPT)
    OUT.mkdir(exist_ok=True)
    (OUT / f"day4_answers_{'v2' if v2 else 'v1'}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    if "--evaluate" in sys.argv:
        evaluate(rows, f"day4-{'v2' if v2 else 'v1'}")
