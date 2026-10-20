"""Northwind (Microsoft sample trading company), 1996-07 .. 1998-05.

Source: https://github.com/graphql-compose/graphql-compose-examples,
examples/northwind/data/csv (MIT).

Adaptation:
* camelCase -> snake_case, ``orderID`` -> ``order_id`` etc., 'NULL' -> missing;
* employee photos / notes / addresses dropped;
* order_details have no time -> ``ts`` = the order's order_date;
* products.unit_price, units_in_stock, units_on_order, discontinued are the
  values at the END of the data (snapshot). The historical price actually
  charged lives in order_details.unit_price -- the two differ for many
  products, which makes the snapshot leak measurable.
"""
import pandas as pd

from lib.common import RAW, snake_columns, to_ts

SRC = RAW / "northwind"


def _read(name):
    df = pd.read_csv(SRC / f"{name}.csv", na_values=["NULL"], keep_default_na=True)
    df = snake_columns(df)
    return df


def prepare():
    orders = _read("orders")
    for c in ["order_date", "required_date", "shipped_date"]:
        orders[c] = to_ts(orders[c])
    orders = orders.drop(columns=["ship_name", "ship_address", "ship_postal_code"])
    details = _read("order_details")
    details["ts"] = details["order_id"].map(orders.set_index("order_id")["order_date"])
    products = _read("products")
    customers = _read("customers")[["customer_id", "city", "region", "country"]]
    employees = _read("employees")[["employee_id", "title", "birth_date", "hire_date", "city",
                                    "country", "reports_to"]]
    for c in ["birth_date", "hire_date"]:
        employees[c] = to_ts(employees[c])
    categories = _read("categories")[["category_id", "category_name"]]
    suppliers = _read("suppliers")[["supplier_id", "city", "country"]]
    shippers = _read("shippers")[["shipper_id", "company_name"]]
    return {"orders": orders, "order_details": details, "products": products,
            "customers": customers, "employees": employees, "categories": categories,
            "suppliers": suppliers, "shippers": shippers}
