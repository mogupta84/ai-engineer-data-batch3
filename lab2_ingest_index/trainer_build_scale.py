"""Day 2, TRAINER ONLY, before class: load the 100,000-vector scale set into both stores, once, for the whole class.

    python lab2_ingest_index/trainer_build_scale.py --search      Azure AI Search index freshcart-scale-100k
    python lab2_ingest_index/trainer_build_scale.py --postgres    pgvector table scale_100k on the class container

--search needs SEARCH_ENDPOINT and SEARCH_ADMIN_KEY in .env (run it once per class search service). --postgres needs
PG_CONNECTION as the trainer (the container's admin login) and PG_STUDENT_PASSWORD, a new password for the students' login. Students only read both, with l2_scale.py, so nobody loads 100,000 rows into a shared server in class.
No model calls: the vectors are made on this laptop, so the only cost is the running container.
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from azure_common import setting
from scale_set import DIM, make_benchmark_set

INDEX, TABLE = "freshcart-scale-100k", "scale_100k"


def build_search(bench):
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration, HnswParameters, SearchField, SearchFieldDataType, SearchIndex, SimpleField,
        VectorSearch, VectorSearchProfile)
    key = AzureKeyCredential(setting("SEARCH_ADMIN_KEY"))
    admin = SearchIndexClient(setting("SEARCH_ENDPOINT"), key)
    if INDEX in admin.list_index_names():
        admin.delete_index(INDEX)
    # The same graph as pgvector's (10 links per point, build effort 100). AI Search accepts a search effort of 100
    # at least, so its ef is 100; the student script also asks it for exhaustive (exact) search.
    admin.create_index(SearchIndex(
        name=INDEX,
        fields=[SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="label", type=SearchFieldDataType.Int32, filterable=True),
                SearchField(name="vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            searchable=True, retrievable=False, stored=False, vector_search_dimensions=DIM,
                            vector_search_profile_name="hnsw")],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-config", parameters=HnswParameters(
                m=10, ef_construction=100, ef_search=100, metric="cosine"))],
            profiles=[VectorSearchProfile(name="hnsw", algorithm_configuration_name="hnsw-config")])))
    client = SearchClient(setting("SEARCH_ENDPOINT"), INDEX, key)
    start = time.time()
    for i in range(0, len(bench.X), 1000):
        batch = [{"id": str(j), "label": int(bench.labels[j]), "vector": bench.X[j].tolist()}
                 for j in range(i, min(i + 1000, len(bench.X)))]
        failed = [r for r in client.upload_documents(batch) if not r.succeeded]
        if failed:
            sys.exit(f"{len(failed)} rows failed at row {i}; first: {failed[0].error_message}")
        if i % 20_000 == 0:
            print(f"  {i:,} rows sent, {time.time() - start:.0f} s")
    print(f"{INDEX}: 100,000 rows sent in {time.time() - start:.0f} s. The count in the portal catches up in a minute.")


def build_postgres(bench):
    import psycopg
    from pgvector.psycopg import register_vector
    password = setting("PG_STUDENT_PASSWORD")                           # checked first, before any slow step
    db = psycopg.connect(setting("PG_CONNECTION"), autocommit=True)
    db.execute("SET search_path = pg_catalog, public")                  # never pick up an object a student made
    database = db.execute("SELECT current_database()").fetchone()[0]
    db.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(db)
    db.execute(f"DROP TABLE IF EXISTS {TABLE}")
    db.execute(f"CREATE TABLE {TABLE} (id integer PRIMARY KEY, label integer, embedding vector({DIM}))")
    start = time.time()
    with db.cursor() as cur, cur.copy(f"COPY {TABLE} (id, label, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
        copy.set_types(["integer", "integer", "vector"])
        for j in range(len(bench.X)):
            copy.write_row([j, int(bench.labels[j]), bench.X[j]])
    print(f"{TABLE}: 100,000 rows copied in {time.time() - start:.0f} s")
    start = time.time()
    db.execute("SET maintenance_work_mem = '1GB'")                      # build the graph in memory, not on disk
    db.execute("SET max_parallel_maintenance_workers = 0")              # a container's shared memory is too small for parallel builds
    db.execute(f"CREATE INDEX ON {TABLE} USING hnsw (embedding vector_cosine_ops) WITH (m = 10, ef_construction = 100)")
    db.execute(f"ANALYZE {TABLE}")
    print(f"HNSW index built in {time.time() - start:.0f} s")
    # Students log in as class_student, never as the trainer: they may read this table and make tables of their own
    # (l2_benchmark.py), but only an owner can drop, empty or re-index a table, so the scale test cannot be broken.
    from psycopg import sql
    exists = db.execute("SELECT 1 FROM pg_roles WHERE rolname = 'class_student'").fetchone()
    db.execute(sql.SQL("{} ROLE class_student WITH LOGIN PASSWORD {}").format(
        sql.SQL("ALTER" if exists else "CREATE"), sql.Literal(password)))
    db.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO class_student").format(sql.Identifier(database)))
    db.execute("GRANT USAGE, CREATE ON SCHEMA public TO class_student")
    db.execute(f"REVOKE ALL ON {TABLE} FROM PUBLIC")
    db.execute(f"GRANT SELECT ON {TABLE} TO class_student")
    print("Student login class_student: may read scale_100k and create its own tables. Give students "
          "PG_CONNECTION=host=<address> port=5432 dbname=labs user=class_student password=<PG_STUDENT_PASSWORD> connect_timeout=10")


if __name__ == "__main__":
    if not {"--search", "--postgres"} & set(sys.argv):
        sys.exit(__doc__)
    bench = make_benchmark_set()
    if "--search" in sys.argv:
        build_search(bench)
    if "--postgres" in sys.argv:
        build_postgres(bench)
