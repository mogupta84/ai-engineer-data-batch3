"""Day 4, code part: write the citation check yourself. No Azure, no cost: it runs on your laptop in a second.

    python lab4_rag_api/l4_checks.py        runs 9 test answers through your three functions and says PASS or FAIL

Then  python lab4_rag_api/l4_score.py --fake  uses YOUR check on 25 real answers, with a made-up source added to some.
Stuck? l4_checks_solution.py is the same file with the gaps filled.

Why in code: the model wrote the citations, so you never ask the model whether they are real. You compare each cited
id with the ids the search actually returned. That comparison is the whole check.
"""
import re

REFUSAL = "I could not find the answer in FreshCart's documents."     # the exact sentence the prompt asks for


# ===================================================== YOUR CODE: three gaps =====================================
def cited_ids(answer):
    r"""GAP 1. Every id the answer cites, as a list, in lower case.
    Ids are 12 letters and digits inside square brackets, like [3f2a9c1d7e44]. One bracket may hold several, separated by
    commas: [3f2a9c1d7e44, 8b0e5a6f1c22]. Words in brackets, like [sic], are not citations.
    Hint: re.findall(r"\[([0-9a-fA-F,\s]+)\]", answer) gives the inside of each bracket; split each on commas."""
    return None


def unknown_citations(cited, retrieved_ids):
    """GAP 2. The cited ids that the search never returned: every id in cited that is not in retrieved_ids."""
    return None


def refusal_reason(answer, cited, unknown):
    """GAP 3. Why this answer must not reach the user, or None if it may. Check in this order:
    the answer starts with the refusal sentence  -> "the model found no answer in the pieces"
    unknown is not empty                          -> "cites pieces the search never returned"
    cited is empty                                -> "no citation"
    otherwise                                     -> None"""
    return "not written yet"
# ==================================================================================================================

TESTS = [
    # (what it shows, the answer, the ids the search returned, the reason you should get)
    ("one real citation", "Up to three claims in thirty days [3f2a9c1d7e44].", ["3f2a9c1d7e44"], None),
    ("two real citations, one bracket", "Fifteen minutes [3F2A9C1D7E44, 8b0e5a6f1c22].",
     ["3f2a9c1d7e44", "8b0e5a6f1c22"], None),
    ("a made-up source", "Three claims [3f2a9c1d7e44] [deadbeef0000].", ["3f2a9c1d7e44"],
     "cites pieces the search never returned"),
    ("a made-up source inside a shared bracket", "Three claims [3f2a9c1d7e44, deadbeef0000].", ["3f2a9c1d7e44"],
     "cites pieces the search never returned"),
    ("no citation at all", "A customer may make up to three refund claims.", ["3f2a9c1d7e44"], "no citation"),
    ("words in brackets are not citations", "Three claims [sic].", ["3f2a9c1d7e44"], "no citation"),
    ("the refusal sentence", REFUSAL, ["3f2a9c1d7e44"], "the model found no answer in the pieces"),
    ("the refusal sentence, then a real citation", REFUSAL + " The policy covers India [3f2a9c1d7e44].",
     ["3f2a9c1d7e44"], "the model found no answer in the pieces"),
    ("real citations from two pieces", "Partner pays [8b0e5a6f1c22]; FreshCart pays [3f2a9c1d7e44].",
     ["3f2a9c1d7e44", "8b0e5a6f1c22", "a1b2c3d4e5f6"], None),
]


def check(answer, retrieved_ids):
    """Your three functions together: what the answer service does with one answer."""
    cited = cited_ids(answer)
    unknown = unknown_citations(cited if cited is not None else [], retrieved_ids)
    return cited, unknown, refusal_reason(answer, cited or [], unknown or [])


def written():
    """True when all three gaps return something real."""
    cited, unknown, reason = check(TESTS[0][1], TESTS[0][2])
    return cited is not None and unknown is not None and reason != "not written yet"


if __name__ == "__main__":
    passed = 0
    for name, answer, retrieved, want in TESTS:
        try:
            cited, unknown, got = check(answer, retrieved)
        except Exception as e:                                   # a bug in a gap: say so and keep going
            cited, unknown, got = None, None, f"error: {type(e).__name__}: {e}"
        ok = got == want and cited is not None and unknown is not None
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:42} you: {got!r}" + ("" if ok else f"   wanted: {want!r}"))
        if not ok and cited is None:
            print("      GAP 1 returned None: return the list of ids")
        elif not ok and unknown is None:
            print("      GAP 2 returned None: return the list of unknown ids")
    print(f"\n{passed} of {len(TESTS)} pass." + (" Now run: python lab4_rag_api/l4_score.py --fake" if passed == len(TESTS)
                                               else " Fix the FAIL lines, then run this again."))
