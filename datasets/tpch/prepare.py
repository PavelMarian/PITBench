"""TPC-H (synthetic order-entry database), scale factor 0.05, 1992 .. 1998.

Source: generated locally with tpchgen-cli (PyPI; the dbgen-compatible
generator from https://github.com/clflushopt/tpchgen-rs):
    tpchgen-cli csv -s 0.05 --output-dir raw/tpch

Why a synthetic database: its temporal semantics are fixed by the TPC-H
specification, so ground truth for several classes is known by construction:
* an order's lines are shipped after the order (ship_date), committed for a
  date (commit_date, planned) and received later (receipt_date);
* l_returnflag, l_linestatus and o_orderstatus are computed against the
  spec's CURRENTDATE = 1995-06-17 -- a snapshot frozen at one fixed date,
  independent of the prediction time;
* customer.c_acctbal is a single current balance (snapshot, no history).

Adaptation: TPC-H prefixes stripped, keys renamed ``*_id`` (o_orderkey ->
order_id ...), comments dropped, lineitem gets ``ts`` = its order's date.
"""
import pandas as pd

from lib.common import RAW, to_ts

SRC = RAW / "tpch"
CURRENTDATE = pd.Timestamp("1995-06-17")

RENAME = {
    "orderkey": "order_id", "custkey": "customer_id", "partkey": "part_id", "suppkey": "supplier_id",
    "nationkey": "nation_id", "regionkey": "region_id", "linenumber": "line_number",
    "orderstatus": "order_status", "totalprice": "total_price", "orderdate": "order_date",
    "orderpriority": "order_priority", "shippriority": "ship_priority", "extendedprice": "extended_price",
    "returnflag": "return_flag", "linestatus": "line_status", "shipdate": "ship_date",
    "commitdate": "commit_date", "receiptdate": "receipt_date", "shipinstruct": "ship_instruct",
    "shipmode": "ship_mode", "acctbal": "acct_balance", "mktsegment": "market_segment",
    "availqty": "avail_qty", "supplycost": "supply_cost", "retailprice": "retail_price",
}


def _read(name):
    df = pd.read_csv(SRC / f"{name}.csv")
    df.columns = [RENAME.get(c.split("_", 1)[1], c.split("_", 1)[1]) for c in df.columns]
    return df.drop(columns=[c for c in ["comment", "name", "address", "phone", "clerk"] if c in df.columns])


def prepare():
    t = {n: _read(n) for n in ["orders", "lineitem", "customer", "part", "supplier", "partsupp",
                               "nation", "region"]}
    t["orders"]["order_date"] = to_ts(t["orders"]["order_date"])
    li = t["lineitem"]
    for c in ["ship_date", "commit_date", "receipt_date"]:
        li[c] = to_ts(li[c])
    li["ts"] = li["order_id"].map(t["orders"].set_index("order_id")["order_date"])
    t["lineitem"] = li
    return t
