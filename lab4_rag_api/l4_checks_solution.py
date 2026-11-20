"""Day 4, optional extra, THE SOLUTION: l4_checks.py with its one line written. Try it yourself first.

WHAT IT IS
    The citation check from lab3_rag_rerank/rag.py (verify), split into three small functions so you can write the
    middle one yourself:
        cited_ids          given: finds every id the answer cites, like [3f2a9c1d7e44]
        unknown_citations  YOUR LINE: the cited ids the search never returned, which are made-up sources
        refusal_reason     given: decides whether the answer may reach the user, and if not, why

WHY IN CODE
    The model wrote the citations, so you never ask the model whether they are real. You compare each cited id with
    the ids the search actually returned. A cited id the search never returned is a made-up source.

HOW TO RUN IT (from the repository folder)
    python lab4_rag_api/l4_checks_solution.py
    It runs 9 test answers through the check and prints PASS or FAIL for each, then "N of 9 pass".
    A test passes only when your line returns exactly the right list AND the reason is right.
    When all 9 pass, two more commands use your check:
        python lab4_rag_api/l4_score.py --fake     your check on 25 real answers, up to 7 of them with a made-up id
        python lab4_rag_api/l4_api.py --my-check   your check decides every reply the model wrote (a question the
                                                   score gate refused stays refused)
    On Windows you can also write the paths with backslashes: python lab4_rag_api\\l4_checks.py
"""
import re

REFUSAL = "I could not find the answer in FreshCart's documents."     # the exact sentence the prompt asks for


def cited_ids(answer):
    """GIVEN, read it: every id the answer cites, in lower case, in the order written.

    Ids sit in square brackets, like [3f2a9c1d7e44]; one bracket may hold several, [3f2a9c1d7e44, 8b0e5a6f1c22].
    Words in brackets, like [sic], are not ids. Example: "Three claims [3F2A9C1D7E44]." gives ["3f2a9c1d7e44"].
    How: re.findall keeps the inside of every bracket that holds only hex letters, digits, commas and spaces;
    each inside is then split on commas."""
    groups = re.findall(r"\[([0-9a-fA-F,\s]+)\]", answer)
    return [i.strip().lower() for group in groups for i in group.split(",") if i.strip()]


# ===================================================== YOUR CODE: one line =======================================
def unknown_citations(cited, retrieved_ids):
    """THE GAP. Return the cited ids that the search never returned: every id in cited that is not in retrieved_ids.

    cited          the ids the answer cites, from cited_ids, for example ["3f2a9c1d7e44", "deadbeef0000"]
    retrieved_ids  the ids of the pieces the search returned, for example ["3f2a9c1d7e44", "8b0e5a6f1c22"]
    return         a list of the cited ids missing from retrieved_ids; here ["deadbeef0000"]. An empty list means
                   every source is real.
    One line is enough; replace return None with it."""
    return [c for c in cited if c not in retrieved_ids]
# ==================================================================================================================


def refusal_reason(answer, cited, unknown):
    """GIVEN, read it: why this answer must not reach the user, or None if it may.

    Checked in this order: the model's own refusal sentence; any made-up source in unknown; no citation at all.
    An answer with real citations and nothing made up returns None, which means it may be shown."""
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
    """The whole check on one answer: returns (cited ids, made-up ids, reason or None). l4_score.py and l4_api.py
    call this when they use your check."""
    cited = cited_ids(answer)
    unknown = unknown_citations(cited if cited is not None else [], retrieved_ids)
    return cited, unknown, refusal_reason(answer, cited or [], unknown or [])


def written():
    """True when the gap returns something other than None, so other scripts know your line is written."""
    cited, unknown, reason = check(TESTS[0][1], TESTS[0][2])
    return cited is not None and unknown is not None


if __name__ == "__main__":
    # Run every test, print PASS or FAIL with the reason, and the total at the end.
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
