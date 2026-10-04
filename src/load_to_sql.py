"""
M1/M2 — load the raw Kaggle CSV into a local SQLite DB and run build_features.sql
to produce the credit_base table used downstream.

Usage:
    python src/load_to_sql.py
"""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "cs-training.csv"
DB_PATH = ROOT / "data" / "processed" / "credit_risk.db"
SQL_FILE = ROOT / "sql" / "build_features.sql"


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_CSV, index_col=0)
    df.index.name = "id"
    df = df.reset_index()

    conn = sqlite3.connect(DB_PATH)
    try:
        df.to_sql("credit_raw", conn, if_exists="replace", index=False)

        sql_script = SQL_FILE.read_text(encoding="utf-8")
        conn.executescript(sql_script)
        conn.commit()

        n_raw = conn.execute("SELECT COUNT(*) FROM credit_raw").fetchone()[0]
        n_base = conn.execute("SELECT COUNT(*) FROM credit_base").fetchone()[0]
        default_rate = conn.execute(
            "SELECT AVG(defaulted) FROM credit_base"
        ).fetchone()[0]

        print(f"Loaded {n_raw} raw rows -> credit_base has {n_base} rows")
        print(f"Default rate: {default_rate:.2%}")
        print(f"DB written to: {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
