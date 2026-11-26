"""Look inside freshcart.db, the data file get_metric reads. Run it from the lab5_agent_tools folder:

    python3 look_inside_db.py      (on a Mac)
    py look_inside_db.py           (on Windows)

It only reads the file. It uses Python's built-in sqlite3, so there is nothing to install.
"""
import pathlib
import sqlite3

DB = pathlib.Path(__file__).parent / "freshcart.db"
db = sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)    # read-only, like the function


def show(title, sql):
    print(f"\n{title}\n  {sql}")
    for row in db.execute(sql).fetchall():
        print("  ", row)


# 1. Which tables are in the file, and how many rows each has
print("1. Tables and their row counts")
for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"):
    count = db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    print(f"   {name:25} {count:6} rows")

# 2. The columns of the three tables get_metric uses
print("\n2. Columns of the tables get_metric uses")
for table in ("orders_v2", "dim_customer", "fact_refund_claim"):
    columns = [c[1] for c in db.execute(f"PRAGMA table_info({table})")]
    print(f"   {table}: {', '.join(columns)}")

# 3. A few rows, to see what the data looks like
show("3. Five orders", "SELECT * FROM orders_v2 ORDER BY order_id LIMIT 5")
show("   The test accounts", "SELECT customer_id, name, city FROM dim_customer WHERE is_internal = 1")

# 4. March 2024 revenue, three ways. Only the last one follows FreshCart's rules, and it is what get_metric returns.
print("\n4. Revenue in March 2024, three ways")
ways = [
    ("every order", "SELECT ROUND(SUM(o.order_value), 2) FROM orders_v2 o "
                    "WHERE o.order_date BETWEEN '2024-03-01' AND '2024-03-31'"),
    ("delivered orders only", "SELECT ROUND(SUM(o.order_value), 2) FROM orders_v2 o "
                              "WHERE o.status = 'delivered' AND o.order_date BETWEEN '2024-03-01' AND '2024-03-31'"),
    ("delivered, test accounts left out (the official number)",
     "SELECT ROUND(SUM(o.order_value), 2) FROM orders_v2 o JOIN dim_customer c ON c.customer_id = o.customer_id "
     "WHERE c.is_internal = 0 AND o.status = 'delivered' AND o.order_date BETWEEN '2024-03-01' AND '2024-03-31'"),
]
for label, sql in ways:
    print(f"   {label:58} {db.execute(sql).fetchone()[0]:>12,.2f}")

db.close()
