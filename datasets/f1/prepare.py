"""Formula 1 (f1db, successor of the Ergast database used by RelBench rel-f1).

Source: https://github.com/f1db/f1db releases, f1db-csv.zip, CC BY 4.0.

Adaptation:
* restricted to seasons >= 2000 (keeps the database small and the sampled
  entities active);
* every key renamed to ``*_id`` (f1db uses ``id``/``raceId``/``driverId``);
* the source stores a race ``date`` (midnight) and, separately, a local
  ``time`` that is missing for many races. Results/standings/pit stops have
  no time of their own. We keep ``race_date`` exactly as the source has it
  (midnight) and add ``available_at``:
    race rows and race-derived rows:  date + time (UTC as given), or
                                      date + 23:59 when the time is missing;
    qualifying rows:                  qualifying date + time if present,
                                      else race date - 1 day + 23:59.
  ``available_at`` is the map's time column. A feature that filters on
  ``race_date <= t`` reads results of a race held later on day ``t`` --
  the boundary class, catchable only because the map uses ``available_at``
  and seeds fall on race days;
* drivers/constructors/circuits keep their career totals (``total_*``,
  ``best_*``): these are computed at dump time over the WHOLE history,
  declared as snapshot columns.
"""
import io
import zipfile

import pandas as pd

from lib.common import RAW, snake_columns, to_ts

SRC = RAW / "f1" / "f1db-csv.zip"
MIN_YEAR = 2000


def _read(z, name):
    return snake_columns(pd.read_csv(io.BytesIO(z.read(f"f1db-{name}.csv")), low_memory=False))


def _at(date, time, default="23:59"):
    t = time.fillna(default).astype(str).str.slice(0, 5)
    return to_ts(date.astype(str) + " " + t)


def prepare():
    z = zipfile.ZipFile(SRC)
    races = _read(z, "races").rename(columns={"id": "race_id", "grand_prix_id": "grand_prix_id"})
    races = races[races["year"] >= MIN_YEAR].copy()
    races["race_date"] = to_ts(races["date"])
    races["available_at"] = _at(races["date"], races["time"])
    q_date = races["qualifying_date"].fillna(
        (races["race_date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"))
    races["qualifying_available_at"] = _at(q_date, races["qualifying_time"])
    keep = ["race_id", "year", "round", "race_date", "available_at", "qualifying_available_at",
            "grand_prix_id", "circuit_id", "circuit_type", "laps", "distance",
            "drivers_championship_decider", "constructors_championship_decider"]
    races = races[keep]
    r = races.set_index("race_id")

    def race_rows(name, avail="available_at"):
        df = _read(z, name)
        df = df[df["race_id"].isin(r.index)].copy()
        df["race_date"] = df["race_id"].map(r["race_date"])
        df["available_at"] = df["race_id"].map(r[avail])
        return df

    results = race_rows("races-race-results")
    results = results[["race_id", "driver_id", "constructor_id", "engine_manufacturer_id",
                       "position_display_order", "position_number", "position_text", "grid_position_number",
                       "laps", "time_millis", "reason_retired", "points", "pit_stops",
                       "fastest_lap", "race_date", "available_at"]]
    qual = race_rows("races-qualifying-results", "qualifying_available_at")
    qual = qual[["race_id", "driver_id", "constructor_id", "position_number", "position_text",
                 "time_millis", "laps", "race_date", "available_at"]]
    dstand = race_rows("races-driver-standings")[
        ["race_id", "driver_id", "position_number", "points", "championship_won", "race_date", "available_at"]]
    cstand = race_rows("races-constructor-standings")[
        ["race_id", "constructor_id", "engine_manufacturer_id", "position_number", "points",
         "championship_won", "race_date", "available_at"]]
    pits = race_rows("races-pit-stops")[
        ["race_id", "driver_id", "constructor_id", "stop", "lap", "time_millis", "race_date", "available_at"]]

    active = set(results["driver_id"]) | set(qual["driver_id"])
    drivers = _read(z, "drivers").rename(columns={"id": "driver_id"})
    drivers = drivers[drivers["driver_id"].isin(active)].copy()
    drivers["date_of_birth"] = to_ts(drivers["date_of_birth"])
    drivers["date_of_death"] = to_ts(drivers["date_of_death"])
    drivers = drivers.drop(columns=["name", "first_name", "last_name", "full_name", "abbreviation",
                                    "place_of_birth", "second_nationality_country_id"])
    cons_active = set(results["constructor_id"])
    constructors = _read(z, "constructors").rename(columns={"id": "constructor_id"})
    constructors = constructors[constructors["constructor_id"].isin(cons_active)]
    constructors = constructors.drop(columns=["name", "full_name"])
    circuits = _read(z, "circuits").rename(columns={"id": "circuit_id"})
    circuits = circuits[["circuit_id", "type", "direction", "country_id", "length", "turns",
                         "total_races_held"]]

    return {"races": races, "race_results": results, "qualifying_results": qual,
            "driver_standings": dstand, "constructor_standings": cstand, "pit_stops": pits,
            "drivers": drivers, "constructors": constructors, "circuits": circuits}
