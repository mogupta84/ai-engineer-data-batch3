"""Day 4, optional extra: score the RAG pipeline on 25 judged questions, the same questions every run.

WHAT IT DOES
    Asks the pipeline in lab3_rag_rerank/rag.py all 25 questions in data/golden/qa_golden.json: 20 that FreshCart's
    documents answer (A01 to A20) and 5 they do not (U01 to U05, U01 is Singapore). For each it prints one line:
        RIGHT                  answerable, and the answer states the expected fact (for example "three")
        WRONG                  answerable, answered, but the expected fact is missing
        refused                answerable, but the pipeline refused (a lost answer)
        refused, as it should  not answerable, and the pipeline refused (the right behaviour)
        ANSWERED               not answerable, but the pipeline answered anyway (a hallucination)
    Then a summary: right facts out of 20, refused though answerable, and refused-as-it-should out of 5.

HOW TO RUN IT (from the repository folder)
    python lab4_rag_api/l4_score.py                    prompt v1, the prompt in rag.py
    python lab4_rag_api/l4_score.py --prompt v2        prompt v2: v1 plus one scope rule (PROMPT_V2 below)
    python lab4_rag_api/l4_score.py --prompt mine      your own prompt, from my_prompt.txt
    python lab4_rag_api/l4_score.py --fake             also plant a made-up id in up to 7 answerable answers and
                                                       count how many the citation check catches (yours too, if
                                                       l4_checks.py is filled in)
    python lab4_rag_api/l4_score.py --evaluate         also Foundry's Groundedness and Relevance evaluators and a
                                                       right-answer judge, all with gpt-4.1-mini (about 70 calls)
    python lab4_rag_api/l4_score.py --evaluate --upload  the same, and the run appears in Foundry (Azure AI Foundry,
                                                       ai.azure.com, where evaluation runs are stored), Evaluations
    Compare runs by their summary lines: change one thing, keep the same 25 questions.
    On Windows you can also write the path with backslashes: python lab4_rag_api\\l4_score.py

HOW IT SCORES
    Rules check facts you can name ("three", "24 hours") with has_fact. Evaluators judge what rules cannot:
    groundedness (is every sentence supported by the pieces?) and relevance (does it answer the question?), each
    1 to 5. The right-answer judge compares each answer with the answer key; it is the one that catches Singapore.
    The judge is gpt-4.1-mini, the same model that writes the answers: a model grading its own kind of work.

COST
    About 25 short answers per run, plus about 70 judge calls with --evaluate: well under $0.10. Each run also uses
    25 reranked searches from the class's shared free allowance, so run it once, not in a loop.
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

# Prompt v2 = prompt v1 plus one rule about scope. It is the fix for U01: the pieces talk about refunds, so v1 answers
# Singapore with India's rules; this rule tells the model to check the country and kind of order first.
PROMPT_V2 = PROMPT + """
Before answering, check that the pieces cover the exact place, product and time the question asks about. FreshCart's
policies cover food delivery orders in India: if the question is about another country or another kind of order,
reply exactly with the sentence above, even when the pieces discuss refunds in general."""


def has_fact(text, expected):
    """True when the answer states one of the expected facts, as a whole word or phrase.

    expected is a list of acceptable forms, for example ["three", "3"]. Citation ids are removed first, so an id
    such as [3f37a93f2727] does not count as the number 3."""
    text = re.sub(r"\[[^\]]*\]", " ", text).lower()
    text = text.replace(chr(8211), "-").replace(chr(8212), "-")      # a range written with a long dash counts as 5-7
    def found(e):                     # short words and numbers match whole; a long stem such as "escalat" may go on
        end = "" if len(e) >= 6 and not e[-1].isdigit() else r"(?![\w]|\.\d)"   # 18 must not match 18.4
        return re.search(r"(?<![\w.])" + re.escape(e.lower()) + end, text)
    return any(found(e) for e in expected)


def option(name, default=None):
    """The value written after --name on the command line, for example "v2" in --prompt v2, or default."""
    i = sys.argv.index(name) if name in sys.argv else -1
    value = sys.argv[i + 1] if 0 <= i < len(sys.argv) - 1 else None
    return default if value is None or value.startswith("--") else value


def my_prompt():
    """Your prompt from my_prompt.txt; lines starting with # are notes to you and are not sent to the model."""
    path = pathlib.Path(__file__).with_name("my_prompt.txt")
    text = "\n".join(line for line in path.read_text(encoding="utf-8-sig").splitlines() if not line.startswith("#")).strip()
    if text == PROMPT.strip():
        sys.exit("my_prompt.txt is still prompt v1. Change one thing in it first, then run this again.")
    if REFUSAL not in text:
        sys.exit(f"my_prompt.txt must keep the refusal sentence word for word: {REFUSAL}\n"
                 "The code recognises a refusal by that exact sentence.")
    return text


def run(client, prompt):
    """Ask all 25 questions with this prompt, print one line each and a summary, and return every result.

    With --fake it also plants [deadbeef0000] in every third answerable question the pipeline did not refuse (up to
    7) and counts how many the citation check catches. If l4_checks.py is filled in, the student's check is run on the same answers and
    compared with the service's check."""
    qa = golden("qa_golden.json")
    rows, fakes, caught = [], 0, 0
    try:                                               # your citation check, from l4_checks.py
        import l4_checks
        mine = l4_checks.written()
    except Exception as e:                             # a bug in a gap must not stop the scoring run
        mine = False
        if "--fake" in sys.argv:
            print(f"Your l4_checks.py raised an error ({type(e).__name__}: {e}); only the service's check runs. "
                  "Run python lab4_rag_api/l4_checks.py to find it.\n")
    mine_caught = agree = compared = 0
    if "--fake" in sys.argv and not mine:
        print("l4_checks.py is not filled in yet, so only the service's own check runs. "
              "Write its one line and run python lab4_rag_api/l4_checks.py first.\n")
    for n, q in enumerate(qa["answerable"] + qa["unanswerable"]):
        answerable = "expect_any" in q
        r = answer(client, q["question"], prompt=prompt)         # the whole pipeline, from rag.py
        right = answerable and not r["refused"] and has_fact(r["answer"], q["expect_any"])
        if "--fake" in sys.argv and answerable and n % 3 == 0 and not r["refused"]:
            # A copy with a made-up source added: the score above and the saved answer stay as the model wrote them.
            faked = verify({**r, "answer": r["answer"] + " [deadbeef0000]"})
            fakes += 1
            caught += "deadbeef0000" in faked["unknown_citations"]
            line = f"{q['id']} made-up source added: service {'CAUGHT' if faked['refused'] else 'NOT CAUGHT'}"
            if mine:
                try:
                    _, unknown, reason = l4_checks.check(faked["answer"], [p["id"] for p in r["retrieved"]])
                    hit = "deadbeef0000" in list(unknown or []) and reason is not None
                except Exception as e:
                    unknown, reason, hit = [], f"error {type(e).__name__}", False
                mine_caught += hit
                line += f", your check {'CAUGHT: ' + str(reason) if hit else 'NOT CAUGHT'}"
            print(line)
        if mine and r["refusal_reason"] != "no sufficiently relevant piece":     # the score gate is not part of your check
            try:
                _, _, reason = l4_checks.check(r["answer"], [p["id"] for p in r["retrieved"]])
            except Exception as e:
                reason = f"error {type(e).__name__}"
            compared += 1
            agree += (reason is None) == (not r["refused"])
        # One word per question, explained at the top of this file.
        verdict = ("RIGHT" if right else "refused" if r["refused"] else "WRONG") if answerable else \
                  ("refused, as it should" if r["refused"] else "ANSWERED")
        print(f"{q['id']} {verdict:22} {r['refusal_reason'] or ''}")
        rows.append({**q, **r, "right": right, "answerable": answerable})
    a = [r for r in rows if r["answerable"]]
    u = [r for r in rows if not r["answerable"]]
    print(f"\nRight facts: {sum(r['right'] for r in a)} of {len(a)}. Refused though answerable: "
          f"{sum(r['refused'] for r in a)} of {len(a)}. Refused, as it should: {sum(r['refused'] for r in u)} of {len(u)}.")
    if fakes:
        print(f"Made-up citations added: {fakes}, caught by the service's check: {caught}"
              + (f", by yours: {mine_caught}" if mine else ""))
    if mine and compared:
        print(f"Your check on the real answers: same verdict as the service's check on {agree} of {compared}.")
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
    """What the answer key says a correct answer must do, in words the judge model can check: for an answerable
    question the right answer, for the others the instruction to say the documents do not answer it."""
    if row["answerable"]:
        return "give this answer from FreshCart's documents, in any words: " + row["expected"]
    return ("say that FreshCart's documents do not answer this question. Giving a policy, a number or advice that "
            "the documents do not state for this exact case is a FAIL, even if it cites a document.")


def evaluate(rows, name):
    """--evaluate: Foundry's Groundedness and Relevance evaluators and the right-answer judge, on every answer that
    was not refused. Runs on your laptop with gpt-4.1-mini as the judge; with --upload the run also appears in the
    Foundry project. Prints the averages and every answer the judge failed."""
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
    # Pick the prompt, run the 25 questions, save every answer to out/day4_answers_<prompt>.json, then score.
    which = option("--prompt") or "v1"
    if which not in ("v1", "v2", "mine"):
        sys.exit("--prompt takes v1, v2 or mine")
    prompt = {"v1": PROMPT, "v2": PROMPT_V2}.get(which) or my_prompt()
    rows = run(models(), prompt)
    OUT.mkdir(exist_ok=True)
    (OUT / f"day4_answers_{which}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    if "--evaluate" in sys.argv:
        evaluate(rows, f"day4-{which}")
