#!/usr/bin/env python3
"""pit_check with an extra map key ``date_cols`` -- a non-invasive wrapper.

pit_check.py parses as datetimes only the columns the map declares as time
channels (time_cols, self_availability_cols). Every other date stays a
string, so feature code doing ``delivered - estimated_delivery`` crashes and
the candidate is reported as ERROR, spending a repair iteration. This
wrapper lets a map list date columns that carry no availability semantics
(planned dates, birth dates, ...) under

    "date_cols": {"<table>": ["<column>", ...]}

so they load as datetimes too. Truncation and canary are unchanged (future
rows' dates are still perturbed like any other value). Usage: exactly the
same arguments as pit_check.py plus --skill-scripts first:

    python -I ext/pit_check_dates.py --skill-scripts <.../pit-repair/scripts> --code ... --db-dir ...
"""
import json
import sys
from pathlib import Path


def main():
    argv = sys.argv[1:]
    i = argv.index("--skill-scripts")
    scripts = argv[i + 1]
    del argv[i:i + 2]
    sys.path.insert(0, scripts)
    import pit_check

    amap_path = argv[argv.index("--availability-map") + 1]
    extra = json.loads(Path(amap_path).read_text(encoding="utf-8")).get("date_cols", {})
    original = pit_check.parse_dates_from_map

    def parse_dates(avail_map):
        out = original(avail_map)
        for table, cols in extra.items():
            out.setdefault(table, [])
            out[table] += [c for c in cols if c not in out[table]]
        return out

    pit_check.parse_dates_from_map = parse_dates
    sys.argv = [str(Path(scripts) / "pit_check.py")] + argv
    pit_check.main()


if __name__ == "__main__":
    main()
