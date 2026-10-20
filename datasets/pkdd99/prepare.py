"""PKDD'99 Financial (Czech bank, "Berka" dataset), 1993-1998.

Source: https://github.com/jlacko/berka-dataset (mirror of the original
PKDD'99 Discovery Challenge .asc files).

Adaptation:
* ';'-separated .asc files -> CSV; ``order`` renamed ``standing_orders``
  (``order`` is an SQL keyword);
* dates stored as YYMMDD integers (card.issued additionally carries a
  constant ' 00:00:00') -> ISO dates (19YY);
* client.birth_number encodes the birth date and the sex (months + 50 for
  women) -> decoded into ``birth_date`` and ``gender``; the raw code is dropped
  because it is the birth date in disguise;
* district columns A1..A16 renamed after the official description. Two of
  them (unemployment rate 1996, crimes 1996) describe the year 1996 and are
  only knowable after it ends -- a snapshot whose validity starts 1997-01-01;
* loan.status is the status at the END of the data (A/B: finished, C/D:
  running) -- a snapshot, the classic PKDD target. It is kept and declared;
* standing_orders have no date at all (state as of the dump) -- declared as
  a snapshot table-wide.
"""
import pandas as pd

from lib.common import RAW

SRC = RAW / "pkdd99"

DISTRICT_COLS = ["district_id", "district_name", "region", "inhabitants",
                 "municipalities_lt_499", "municipalities_500_1999", "municipalities_2000_9999",
                 "municipalities_gt_10000", "cities", "urban_ratio", "average_salary",
                 "unemployment_1995", "unemployment_1996", "entrepreneurs_per_1000",
                 "crimes_1995", "crimes_1996"]


def _read(name):
    return pd.read_csv(SRC / f"{name}.asc", sep=";", low_memory=False)


def _yymmdd(s):
    s = s.astype(str).str.slice(0, 6)
    return pd.to_datetime("19" + s, format="%Y%m%d", errors="coerce")


def _birth(code):
    code = code.astype(str).str.zfill(6)
    yy = code.str.slice(0, 2).astype(int)
    mm = code.str.slice(2, 4).astype(int)
    dd = code.str.slice(4, 6).astype(int)
    female = mm > 50
    mm = mm.where(~female, mm - 50)
    date = pd.to_datetime(dict(year=1900 + yy, month=mm, day=dd), errors="coerce")
    return date, female.map({True: "F", False: "M"})


def prepare():
    account = _read("account").rename(columns={"date": "open_date"})
    account["open_date"] = _yymmdd(account["open_date"])
    client = _read("client")
    client["birth_date"], client["gender"] = _birth(client["birth_number"])
    client = client.drop(columns=["birth_number"])
    disp = _read("disp")
    card = _read("card")
    card["issued"] = _yymmdd(card["issued"])
    loan = _read("loan").rename(columns={"date": "granted_date"})
    loan["granted_date"] = _yymmdd(loan["granted_date"])
    orders = _read("order")
    trans = _read("trans")
    trans["date"] = _yymmdd(trans["date"])
    trans = trans.rename(columns={"account": "partner_account", "bank": "partner_bank"})
    district = _read("district")
    district.columns = DISTRICT_COLS
    for c in ["unemployment_1995", "crimes_1995"]:
        district[c] = pd.to_numeric(district[c], errors="coerce")   # '?' marks missing
    district["stats_1996_known_from"] = pd.Timestamp("1997-01-01")

    return {"account": account, "client": client, "disp": disp, "card": card, "loan": loan,
            "standing_orders": orders, "trans": trans, "district": district}
