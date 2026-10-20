"""AdventureWorks OLTP (Microsoft sample database), Sales + Production subset.

Source: https://github.com/microsoft/sql-server-samples,
samples/databases/adventure-works/oltp-install-script/*.csv (MIT).

Adaptation:
* tab-separated, header-less files -> CSV with snake_case names taken from
  the table definitions in instawdb.sql (column counts asserted);
* ``rowguid`` and free-text columns dropped; ``business_entity_id`` of
  sales_person renamed ``sales_person_id``;
* sales_order_detail has no time of its own -> ``ts`` = header order_date;
* product.list_price / standard_cost are CURRENT values (snapshot). The
  source also has the true history (product_list_price_history,
  product_cost_history). We add ``list_price_as_of`` / ``standard_cost_as_of``
  = start date of the price currently stored, so the snapshot channel knows
  from when the stored value is valid;
* ``modified_date`` is kept everywhere: it is re-stamped at load time (e.g.
  2025 for products whose orders end in 2025) -- a trap for anyone who
  takes it for the row's time.
"""
import pandas as pd

from lib.common import RAW, to_ts

SRC = RAW / "adventureworks"

COLS = {
    "SalesOrderHeader": ["sales_order_id", "revision_number", "order_date", "due_date", "ship_date",
                         "status", "online_order_flag", "sales_order_number", "purchase_order_number",
                         "account_number", "customer_id", "sales_person_id", "territory_id",
                         "bill_to_address_id", "ship_to_address_id", "ship_method_id", "credit_card_id",
                         "credit_card_approval_code", "currency_rate_id", "sub_total", "tax_amt",
                         "freight", "total_due", "comment", "rowguid", "modified_date"],
    "SalesOrderDetail": ["sales_order_id", "sales_order_detail_id", "carrier_tracking_number",
                         "order_qty", "product_id", "special_offer_id", "unit_price",
                         "unit_price_discount", "line_total", "rowguid", "modified_date"],
    "Customer": ["customer_id", "person_id", "store_id", "territory_id", "account_number",
                 "rowguid", "modified_date"],
    "Product": ["product_id", "name", "product_number", "make_flag", "finished_goods_flag", "color",
                "safety_stock_level", "reorder_point", "standard_cost", "list_price", "size",
                "size_unit_measure_code", "weight_unit_measure_code", "weight", "days_to_manufacture",
                "product_line", "class", "style", "product_subcategory_id", "product_model_id",
                "sell_start_date", "sell_end_date", "discontinued_date", "rowguid", "modified_date"],
    "ProductListPriceHistory": ["product_id", "start_date", "end_date", "list_price", "modified_date"],
    "ProductCostHistory": ["product_id", "start_date", "end_date", "standard_cost", "modified_date"],
    "ProductSubcategory": ["product_subcategory_id", "product_category_id", "name", "rowguid",
                           "modified_date"],
    "ProductCategory": ["product_category_id", "name", "rowguid", "modified_date"],
    "SpecialOffer": ["special_offer_id", "description", "discount_pct", "type", "category",
                     "start_date", "end_date", "min_qty", "max_qty", "rowguid", "modified_date"],
    "SalesTerritory": ["territory_id", "name", "country_region_code", "group", "sales_ytd",
                       "sales_last_year", "cost_ytd", "cost_last_year", "rowguid", "modified_date"],
    "SalesPerson": ["sales_person_id", "territory_id", "sales_quota", "bonus", "commission_pct",
                    "sales_ytd", "sales_last_year", "rowguid", "modified_date"],
}
DATES = {"order_date", "due_date", "ship_date", "sell_start_date", "sell_end_date",
         "discontinued_date", "start_date", "end_date", "modified_date"}
DROP = {"rowguid", "comment", "credit_card_approval_code", "carrier_tracking_number",
        "purchase_order_number", "sales_order_number", "account_number", "description", "name"}
NAMES = {"SalesOrderHeader": "sales_order_header", "SalesOrderDetail": "sales_order_detail",
         "Customer": "customer", "Product": "product",
         "ProductListPriceHistory": "product_list_price_history",
         "ProductCostHistory": "product_cost_history", "ProductSubcategory": "product_subcategory",
         "ProductCategory": "product_category", "SpecialOffer": "special_offer",
         "SalesTerritory": "sales_territory", "SalesPerson": "sales_person"}


def _read(name):
    df = pd.read_csv(SRC / f"{name}.csv", sep="\t", header=None, quoting=3, dtype=str,
                     keep_default_na=False, encoding="latin-1")
    cols = COLS[name]
    assert df.shape[1] == len(cols), f"{name}: {df.shape[1]} columns in file, {len(cols)} expected"
    df.columns = cols
    df = df.replace("", pd.NA)
    for c in cols:
        if c in DATES:
            df[c] = to_ts(df[c])
        elif c not in DROP and c not in {"color", "size", "size_unit_measure_code",
                                         "weight_unit_measure_code", "product_line", "class", "style",
                                         "product_number", "type", "category", "group",
                                         "country_region_code"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = [c for c in cols if c not in DROP or (name == "SalesTerritory" and c == "name")]
    return df[keep]


def prepare():
    t = {NAMES[n]: _read(n) for n in COLS}
    hdr = t["sales_order_header"].set_index("sales_order_id")
    det = t["sales_order_detail"]
    det["ts"] = det["sales_order_id"].map(hdr["order_date"])
    prod = t["product"]
    for snap, hist in [("list_price", "product_list_price_history"),
                       ("standard_cost", "product_cost_history")]:
        h = t[hist]
        current = h[h["end_date"].isna()].groupby("product_id")["start_date"].max()
        latest = h.groupby("product_id")["start_date"].max()
        prod[f"{snap}_as_of"] = prod["product_id"].map(current).fillna(prod["product_id"].map(latest))
    return t
