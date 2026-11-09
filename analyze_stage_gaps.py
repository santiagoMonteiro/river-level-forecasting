#!/usr/bin/env python3
"""Report consecutive missing daily river-stage observations for each station."""

import csv
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
START_DATE = date(1991, 1, 1)
END_DATE = date(2023, 12, 31)
ONE_DAY = timedelta(days=1)


def observed_dates(path):
    """Return dates with at least one numeric stage at consistency level 2."""
    dates = set()
    with path.open(encoding="latin-1", newline="") as source:
        for line in source:
            if line.startswith("EstacaoCodigo;NivelConsistencia;"):
                reader = csv.DictReader(source, fieldnames=line.strip().split(";"), delimiter=";")
                break
        else:
            raise ValueError(f"Stage header not found in {path}")
        columns = [f"Cota{day:02d}" for day in range(1, 32)]
        if not {"Data", "NivelConsistencia", *columns}.issubset(reader.fieldnames):
            raise ValueError(f"Missing date, consistency or stage columns in {path}")
        for row in reader:
            if (row.get("NivelConsistencia") or "").strip() != "2":
                continue
            reference = datetime.strptime(row["Data"].strip(), "%d/%m/%Y")
            for day, column in enumerate(columns, start=1):
                value = (row.get(column) or "").strip()
                if not value:
                    continue
                try:
                    stage = float(value.replace(",", "."))
                    observation_date = date(reference.year, reference.month, day)
                except ValueError:
                    continue
                if stage == stage and abs(stage) != float("inf"):
                    dates.add(observation_date)
    return dates


def missing_intervals(observed, start, end):
    """Yield inclusive intervals without observations."""
    current = start
    gap_start = None
    while current <= end:
        if current not in observed:
            if gap_start is None:
                gap_start = current
        elif gap_start is not None:
            yield gap_start, current - ONE_DAY
            gap_start = None
        current += ONE_DAY
    if gap_start is not None:
        yield gap_start, end


def main():
    if START_DATE > END_DATE:
        raise ValueError("START_DATE must be earlier than or equal to END_DATE")
    if not DATA_DIR.is_dir():
        raise ValueError(f"Data directory not found: {DATA_DIR}")
    files = sorted(DATA_DIR.rglob("*_Cotas.csv"))
    if not files:
        raise ValueError(f"No *_Cotas.csv files found in {DATA_DIR}")
    print(f"Analysis period: {START_DATE} to {END_DATE} (inclusive)")
    print("Criterion: at least one numeric stage with consistency level 2 per day.")
    for path in files:
        gaps = list(missing_intervals(observed_dates(path), START_DATE, END_DATE))
        total = sum((end - start).days + 1 for start, end in gaps)
        print(f"\nStation: {path.parent.name} ({path.stem.removesuffix('_Cotas')})")
        print(f"  Missing days: {total}; intervals: {len(gaps)}")
        if not gaps:
            print("  No gaps found.")
        for start, end in gaps:
            print(f"  {start} to {end} — {(end - start).days + 1} day(s)")


if __name__ == "__main__":
    main()
