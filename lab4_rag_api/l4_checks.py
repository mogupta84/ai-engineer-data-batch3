"""Day 4, code part: the citation check. You write ONE line of it. No Azure, no cost: it runs on your laptop in a second.

    python lab4_rag_api/l4_checks.py        runs 9 test answers through the check and says PASS or FAIL

Why in code: the model wrote the citations, so you never ask the model whether they are real. You compare each cited
id with the ids the search actually returned. A cited id the search never returned is a made-up source.
Stuck? l4_checks_solution.py is the same file with the line written.
"""
import re

REFUSAL = "I could not find the answer in FreshCart's documents."     # the exact sentence the prompt asks for


def cited_ids(answer):
    """GIVEN, read it: every id the answer cites, in lower case. Ids sit in square brackets, like [3f2a9c1d7e44];
    one bracket may hold several, [3f2a9c1d7e44, 8b0e5a6f1c22]. Words in brackets, like [sic], are not ids."""
    groups = re.findall(r"\[([0-9a-fA-F,\s]+)\]", answer)
    return [i.strip().lower() for group in groups for i in group.split(",") if i.strip()]


# ===================================================== YOUR CODE: one line =======================================
def unknown_citations(cited, retrieved_ids):
    """THE GAP. Return the cited ids that the search never returned: every id in cited that is not in retrieved_ids.
    Both are lists of ids. One line is enough; replace return None with it."""
    return None
# ==================================================================================================================


def refusal_reason(answer, cited, unknown):
    """GIVEN, read it: why this answer must not reach the user, or None if it may."""
    if answer.startswith(REFUSAL[:30]):
        return "the model found no answer in the pieces"
    if unknown:
        return "cites pieces the search never returned"
    if not cited:
        return "no citation"
    return None


TESTS = [
    # (what it shows, the answer, the ids the search returned, the reason you should get, the made-up ids)
    ("one real citation", "Up to three claims in thirty days [3f2a9c1d7e44].", ["3f2a9c1d7e44"], None, []),
    ("two real citations, one bracket", "Fifteen minutes [3F2A9C1D7E44, 8b0e5a6f1c22].",
     ["3f2a9c1d7e44", "8b0e5a6f1c22"], None, []),
    ("two made-up sources", "Three claims [3f2a9c1d7e44] [deadbeef0000] [0123456789ab].", ["3f2a9c1d7e44"],
     "cites pieces the search never returned", ["deadbeef0000", "0123456789ab"]),
    ("a made-up source inside a shared bracket", "Three claims [3f2a9c1d7e44, deadbeef0000].", ["3f2a9c1d7e44"],
     "cites pieces the search never returned", ["deadbeef0000"]),
    ("no citation at all", "A customer may make up to three refund claims.", ["3f2a9c1d7e44"], "no citation", []),
    ("words in brackets are not citations", "Three claims [sic].", ["3f2a9c1d7e44"], "no citation", []),
    ("the refusal sentence", REFUSAL, ["3f2a9c1d7e44"], "the model found no answer in the pieces", []),
    ("the refusal sentence, then a real citation", REFUSAL + " The policy covers India [3f2a9c1d7e44].",
     ["3f2a9c1d7e44"], "the model found no answer in the pieces", []),
    ("real citations from two pieces", "Partner pays [8b0e5a6f1c22]; FreshCart pays [3f2a9c1d7e44].",
     ["3f2a9c1d7e44", "8b0e5a6f1c22", "a1b2c3d4e5f6"], None, []),
]


def check(answer, retrieved_ids):
    """The whole check: what the answer service does with one answer."""
    cited = cited_ids(answer)
    unknown = unknown_citations(cited if cited is not None else [], retrieved_ids)
    return cited, unknown, refusal_reason(answer, cited or [], unknown or [])


def written():
    """True when the gap returns something real."""
    cited, unknown, reason = check(TESTS[0][1], TESTS[0][2])
    return cited is not None and unknown is not None


if __name__ == "__main__":
    passed = 0
    for name, answer, retrieved, want, want_unknown in TESTS:
        try:
            cited, unknown, got = check(answer, retrieved)
        except Exception as e:                                   # a bug in a gap: say so and keep going
            cited, unknown, got = None, None, f"error: {type(e).__name__}: {e}"
        same_list = isinstance(unknown, list) and sorted(unknown) == sorted(want_unknown)
        ok = got == want and same_list
        passed += ok
        if unknown is None:
            print(f"FAIL  {name:42} the gap still returns None")
            continue
        if not same_list:
            print(f"FAIL  {name:42} your line returned {unknown!r}, wanted the list {want_unknown!r}")
            continue
        print(f"{'PASS' if ok else 'FAIL'}  {name:42} reason: {got!r}" + ("" if ok else f"   wanted: {want!r}"))
    print(f"\n{passed} of {len(TESTS)} pass." + (" Now run: python lab4_rag_api/l4_score.py --fake" if passed == len(TESTS)
                                               else " Fix the FAIL lines, then run this again."))
