"""Olist (Brazilian e-commerce) -- the corpus PITBULL was validated on.

Source: the copy kept in this project's history (commit a9cd898,
PITFALL_olist_data/), originally Kaggle "olist/brazilian-ecommerce", CC BY-NC-SA 4.0.

Adaptation:
* table names shortened to the ones used by the existing Olist map
  (orders, order_items, payments, reviews, customers, products, sellers);
* order_items and payments have NO time of their own in the source; both get
  ``ts`` = the parent order's purchase timestamp (time inheritance -- the
  same choice the published PITBULL numbers were measured under);
* payments additionally get ``approved_at`` = the parent order's approval
  time, so the stricter availability reading (payment known on approval)
  is available as a map variant;
* ``order_status_events``: a derived event table (order_id, status, ts)
  reconstructed from the order's timestamps, so the status *as of t* is
  computable -- the source only stores the final status (a snapshot).
"""
import pandas as pd

from lib.common import RAW, to_ts

SRC = RAW / "olist"


def prepare():
    orders = pd.read_csv(SRC / "olist_orders_dataset.csv")
    for c in ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
              "order_delivered_customer_date", "order_estimated_delivery_date"]:
        orders[c] = to_ts(orders[c])
    items = pd.read_csv(SRC / "olist_order_items_dataset.csv")
    items["shipping_limit_date"] = to_ts(items["shipping_limit_date"])
    pays = pd.read_csv(SRC / "olist_order_payments_dataset.csv")
    reviews = pd.read_csv(SRC / "olist_order_reviews_dataset.csv")
    for c in ["review_creation_date", "review_answer_timestamp"]:
        reviews[c] = to_ts(reviews[c])
    customers = pd.read_csv(SRC / "olist_customers_dataset.csv")
    products = pd.read_csv(SRC / "olist_products_dataset.csv")
    sellers = pd.read_csv(SRC / "olist_sellers_dataset.csv")

    o = orders.set_index("order_id")
    items["ts"] = items["order_id"].map(o["order_purchase_timestamp"])
    pays["ts"] = pays["order_id"].map(o["order_purchase_timestamp"])
    pays["approved_at"] = pays["order_id"].map(o["order_approved_at"])
    # payment_sequential/installments are counters, not keys

    ev = []
    for col, status in [("order_purchase_timestamp", "created"), ("order_approved_at", "approved"),
                        ("order_delivered_carrier_date", "shipped"),
                        ("order_delivered_customer_date", "delivered")]:
        e = orders.loc[orders[col].notna(), ["order_id", col]].rename(columns={col: "ts"})
        e["status"] = status
        ev.append(e)
    events = pd.concat(ev, ignore_index=True)[["order_id", "status", "ts"]]
    events = events.sort_values(["ts", "order_id"]).reset_index(drop=True)

    tables = {"orders": orders, "order_items": items, "payments": pays, "reviews": reviews,
              "customers": customers, "products": products, "sellers": sellers,
              "order_status_events": events}
    return tables
