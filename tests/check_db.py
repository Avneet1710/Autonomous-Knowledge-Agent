import sqlite3, os

for db_name, db_path in [("cultpass.db", "data/external/cultpass.db"), ("udahub.db", "data/core/udahub.db")]:
    size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    print(f"{db_name}  ({size} bytes)")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for (table,) in c.fetchall():
        c.execute(f'SELECT COUNT(*) FROM "{table}"')
        print(f"  {table}: {c.fetchone()[0]} rows")
    conn.close()
    print()
