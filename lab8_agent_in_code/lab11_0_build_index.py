"""Lab 11, trainer only, once before class: the shared search index for G2, with an access label on every piece.

Each piece of the 7 FreshCart files (as Document Intelligence read them on Day 2) gets an 'audience' label: who may read it.
The policy, the service levels and the procedure are for all staff; the partner agreement (who pays for a refund)
for finance; the March claims register (customer IDs) for support leads; the quarterly review for managers.

    python lab11_0_build_index.py      needs SEARCH_ENDPOINT and SEARCH_ADMIN_KEY (the trainer's admin key) in .env
"""
import os                      # read the settings
import pathlib                 # find the Day 2 read files
import re                      # cut each file at its headings
import sys                     # stop with a clear message

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchableField, SearchFieldDataType

import lab8_common                                     # reads the .env file in this folder

INDEX = "freshcart-docs-lab11"
HERE = pathlib.Path(__file__).parent
READ = next((p for p in (HERE.parent / "out" / "read", HERE.parent.parent / "out" / "read") if p.exists()),
            None)                                       # what lab2_ingest_index/trainer_build_shared_index.py read (Day 2)
AUDIENCE = {
    "refund_policy_v3_2024.pdf": ["all"],
    "refund_policy_v2_2022.pdf": ["all"],
    "support_sla_matrix.docx": ["all"],
    "support_escalation_sop_scanned.pdf": ["all"],
    "partner_agreement_extract.pdf": ["finance"],
    "refund_claims_march_2024.xlsx": ["support_leads"],
    "support_qbr_q1_2024.pptx": ["managers"],
}


def pieces():
    """One piece per section of each file, cut at its '## ' headings, with the file's audience label."""
    for path in sorted(READ.glob("*.md")):
        file = path.name[:-3]
        parts = re.split(r"\n(?=## )", path.read_text(encoding="utf-8"))
        for n, part in enumerate(p for p in parts if p.strip()):
            title = part.strip().splitlines()[0].lstrip("# ").strip()
            yield {"id": f"{re.sub(r'[^A-Za-z0-9]', '_', file)}_{n}", "file": file, "title": title,
                   "content": part.strip(), "audience": AUDIENCE[file]}


def main():
    endpoint, key = os.environ.get("SEARCH_ENDPOINT"), os.environ.get("SEARCH_ADMIN_KEY")
    if not (endpoint and key and READ):
        sys.exit("Needs SEARCH_ENDPOINT and SEARCH_ADMIN_KEY in .env, and the out/read folder (run lab2_ingest_index/trainer_build_shared_index.py first).")
    fields = [SimpleField(name="id", type=SearchFieldDataType.String, key=True),
              SimpleField(name="file", type=SearchFieldDataType.String, filterable=True),
              SearchableField(name="title", type=SearchFieldDataType.String),
              SearchableField(name="content", type=SearchFieldDataType.String),
              SimpleField(name="audience", type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                          filterable=True)]                          # who may read it: the field the filter uses
    admin = SearchIndexClient(endpoint, AzureKeyCredential(key))
    admin.create_or_update_index(SearchIndex(name=INDEX, fields=fields))
    items = list(pieces())
    SearchClient(endpoint, INDEX, AzureKeyCredential(key)).upload_documents(items)
    for file in AUDIENCE:
        print(f"{file:<38} {sum(i['file'] == file for i in items):>2} pieces   for {', '.join(AUDIENCE[file])}")
    print(f"Index {INDEX}: {len(items)} pieces.")


if __name__ == "__main__":
    main()
