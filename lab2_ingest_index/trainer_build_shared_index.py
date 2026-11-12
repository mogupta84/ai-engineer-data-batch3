"""Day 2, TRAINER ONLY, before class: build the shared index freshcart-chunks that the Day 2 to 4 code parts read.

    python lab2_ingest_index/trainer_build_shared_index.py            build it on SEARCH_ENDPOINT
    python lab2_ingest_index/trainer_build_shared_index.py --reuse    skip Document Intelligence; reuse out/read/

Run it once per class search service (change SEARCH_ENDPOINT and SEARCH_ADMIN_KEY in .env between runs).
It reads the 7 files with Document Intelligence, cuts them into pieces of 500 characters (50 shared with the next),
labels each piece with its file's type, version, date and current-or-not, embeds them with text-embedding-3-small
and stores them. Cost: 9 pages of Document Intelligence and about 60 short embeddings, under $0.15.

Students get only the query key, which can read but not change the index.
"""
import hashlib        # a short, stable id for each piece
import pathlib        # find the read files
import re             # take out the table markup once its rows are written as lines
import sys            # read --reuse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from azure_common import LABELS, OUT, RAW, SHARED_INDEX, embed, models, setting, sign_in

READ = OUT / "read"


def table_rows(table):
    """A table as one line per row, each cell named by its column, so a row read alone still says what it means."""
    grid = {}
    for cell in table.cells:
        grid.setdefault(cell.row_index, {})[cell.column_index] = cell.content.strip()
    header = grid.get(0, {})
    return ["; ".join(f"{header.get(c, '')}: {v}" for c, v in sorted(row.items())) for r, row in sorted(grid.items()) if r > 0]


def read_all():
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import DocumentContentFormat
    client = DocumentIntelligenceClient(setting("DOCINTEL_ENDPOINT"), sign_in())
    READ.mkdir(parents=True, exist_ok=True)
    for path in sorted(RAW.iterdir()):
        result = client.begin_analyze_document("prebuilt-layout", body=path.read_bytes(),
                                               output_content_format=DocumentContentFormat.MARKDOWN).result()
        text = result.content
        if result.tables:
            text = re.sub(r"<table>.*?</table>", "", text, flags=re.S)
            for table in result.tables:
                text += "\n\n" + "\n".join(table_rows(table))
        (READ / (path.name + ".md")).write_text(text, encoding="utf-8")
        print(f"read {path.name:38} {len(text):5} characters")


def pieces():
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    cutter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    out = []
    for name, labels in LABELS.items():
        for text in cutter.split_text((READ / (name + ".md")).read_text(encoding="utf-8")):
            out.append({"id": hashlib.sha256(f"{name}|{text}".encode()).hexdigest()[:12], "file": name,
                        "content": text, **labels})
    return out


def definition():
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration, SearchableField, SearchField, SearchFieldDataType, SearchIndex, SemanticConfiguration,
        SemanticField, SemanticPrioritizedFields, SemanticSearch, SimpleField, VectorSearch, VectorSearchProfile)
    return SearchIndex(
        name=SHARED_INDEX,
        fields=[SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="file", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SimpleField(name="doc_type", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="version", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="effective_date", type=SearchFieldDataType.String, filterable=True, sortable=True),
                SimpleField(name="current", type=SearchFieldDataType.Boolean, filterable=True),
                SearchField(name="vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            searchable=True, retrievable=True, vector_search_dimensions=1536,
                            vector_search_profile_name="meaning")],
        vector_search=VectorSearch(algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
                                   profiles=[VectorSearchProfile(name="meaning", algorithm_configuration_name="hnsw")]),
        semantic_search=SemanticSearch(configurations=[SemanticConfiguration(
            name="rerank", prioritized_fields=SemanticPrioritizedFields(content_fields=[SemanticField(field_name="content")]))]))


def main():
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    if "--reuse" not in sys.argv or not READ.exists():
        read_all()
    items = pieces()
    for item, vector in zip(items, embed(models(), [p["content"] for p in items])):
        item["vector"] = vector
    key = AzureKeyCredential(setting("SEARCH_ADMIN_KEY"))
    admin = SearchIndexClient(setting("SEARCH_ENDPOINT"), key)
    if SHARED_INDEX in admin.list_index_names():
        admin.delete_index(SHARED_INDEX)                       # a rebuild starts clean
    admin.create_index(definition())
    done = SearchClient(setting("SEARCH_ENDPOINT"), SHARED_INDEX, key).upload_documents(items)
    print(f"\n{SHARED_INDEX} on {setting('SEARCH_ENDPOINT')}: {sum(r.succeeded for r in done)} of {len(items)} pieces stored")
    for name in LABELS:
        print(f"  {name:38} {sum(p['file'] == name for p in items)} pieces")


main()
