"""Sakila (MySQL sample DVD-rental database, SQLite port), 2005-05 .. 2006-02.

Source: https://github.com/jOOQ/sakila, sqlite-sakila-db (BSD).

Adaptation:
* the SQLite schema + inserts are loaded into an in-memory SQLite database
  with the schema's triggers removed (they would overwrite ``last_update``
  with the load time) and exported table by table;
* personal data (names, e-mail, address, phone, pictures) dropped;
* film text columns dropped; film_actor/actor omitted;
* customer.create_date is 2006-02-14 for EVERY customer -- i.e. after all
  of their rentals. It is not a usable existence time and is deliberately
  NOT in the map (entity-existence class); first rental is the only
  existence evidence;
* ``last_update`` = 2006-02-15 nearly everywhere: a load-time re-stamp;
* payment.amount includes late fees that depend on the return, yet the
  payment row is stamped at rental time (a late value re-stamped early).
"""
import re
import sqlite3

import pandas as pd

from lib.common import RAW, to_ts

SRC = RAW / "sakila"
KEEP = {
    "rental": ["rental_id", "rental_date", "inventory_id", "customer_id", "return_date", "staff_id",
               "last_update"],
    "payment": ["payment_id", "customer_id", "staff_id", "rental_id", "amount", "payment_date",
                "last_update"],
    "inventory": ["inventory_id", "film_id", "store_id", "last_update"],
    "film": ["film_id", "release_year", "language_id", "rental_duration", "rental_rate", "length",
             "replacement_cost", "rating", "last_update"],
    "film_category": ["film_id", "category_id", "last_update"],
    "category": ["category_id", "name", "last_update"],
    "customer": ["customer_id", "store_id", "active", "create_date", "last_update"],
    "store": ["store_id", "manager_staff_id", "last_update"],
    "staff": ["staff_id", "store_id", "active", "last_update"],
}
DATES = {"rental_date", "return_date", "payment_date", "create_date", "last_update"}


def prepare():
    schema = (SRC / "schema.sql").read_text(encoding="utf-8", errors="replace")
    schema = re.sub(r"CREATE TRIGGER.*?END\s*;", "", schema, flags=re.S | re.I)
    data = (SRC / "data.sql").read_text(encoding="utf-8", errors="replace")
    con = sqlite3.connect(":memory:")
    con.executescript(schema)
    con.executescript(data)
    out = {}
    for t, cols in KEEP.items():
        df = pd.read_sql_query(f"SELECT {', '.join(cols)} FROM {t}", con)
        for c in cols:
            if c in DATES:
                df[c] = to_ts(df[c])
        out[t] = df
    con.close()
    out["store"] = out["store"].rename(columns={"manager_staff_id": "manager_staff_id"})
    return out
