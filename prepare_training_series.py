#!/usr/bin/env python3
"""Prepare daily river-stage series for training.

Fill Humaitá with Santo Antônio observations, then interpolate remaining
internal gaps linearly, following the notebook. Santo Antônio is only a donor.
Requires pandas: python3 -m pip install pandas
"""

from datetime import date
from pathlib import Path
import sys

import pandas as pd

# Edit these settings to change the input, output, or inclusive analysis period.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
START_DATE = date(1991, 1, 1)
END_DATE = date(2023, 12, 31)
HUMAITA_CODE = "15630000"
SANTO_ANTONIO_CODE = "15629000"
STAGE_COLUMNS = [f"Cota{day:02d}" for day in range(1, 32)]


def load_series(path):
    """Convert consistency level 2 monthly records into daily observations."""
    with path.open(encoding="latin-1") as source:
        for header_line, line in enumerate(source):
            if line.startswith("EstacaoCodigo;NivelConsistencia;"):
                break
        else:
            raise ValueError(f"Header not found in {path}")
    data = pd.read_csv(path, sep=";", encoding="latin-1", skiprows=header_line,
                       dtype=str, low_memory=False)
    missing = {"Data", "NivelConsistencia", *STAGE_COLUMNS}.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
    data = data.loc[data["NivelConsistencia"].str.strip() == "2"]
    long = data.melt(id_vars="Data", value_vars=STAGE_COLUMNS,
                     var_name="day", value_name="river_stage")
    reference = pd.to_datetime(long["Data"], format="%d/%m/%Y", errors="coerce")
    if reference.isna().any():
        raise ValueError(f"Invalid reference date in {path}")
    long["date"] = pd.to_datetime({"year": reference.dt.year,
                                   "month": reference.dt.month,
                                   "day": long["day"].str[4:].astype(int)}, errors="coerce")
    long["river_stage"] = pd.to_numeric(
        long["river_stage"].str.replace(",", ".", regex=False), errors="coerce"
    )
    long = long.dropna(subset=["date", "river_stage"])
    long = long[long["river_stage"].abs() != float("inf")]
    # Keep the first available observation for each day, as in model preparation.
    long = long.drop_duplicates(subset="date", keep="first").sort_values("date")
    return long.set_index("date")["river_stage"]


def main():
    if START_DATE > END_DATE:
        raise ValueError("START_DATE must be earlier than or equal to END_DATE")
    files = sorted(DATA_DIR.rglob("*_Cotas.csv"))
    by_code = {}
    for path in files:
        code = path.name.removesuffix("_Cotas.csv")
        if code in by_code:
            raise ValueError(f"Multiple files found for station {code}")
        by_code[code] = path
    for code in (HUMAITA_CODE, SANTO_ANTONIO_CODE):
        if code not in by_code:
            raise ValueError(f"Stage file for station {code} not found in {DATA_DIR}")

    calendar = pd.date_range(START_DATE, END_DATE, freq="D", name="date")
    donor = load_series(by_code[SANTO_ANTONIO_CODE]).reindex(calendar)
    outputs = []
    for code, path in by_code.items():
        if code == SANTO_ANTONIO_CODE:
            continue
        series = load_series(path).reindex(calendar)
        originally_missing = int(series.isna().sum())
        filled = interpolated = 0
        if code == HUMAITA_CODE:
            filled = int((series.isna() & donor.notna()).sum())
            series = series.fillna(donor)
            before = int(series.isna().sum())
            series = series.interpolate(method="linear", limit_area="inside")
            interpolated = before - int(series.isna().sum())
        remaining = int(series.isna().sum())
        if remaining:
            examples = ", ".join(series.index[series.isna()].strftime("%Y-%m-%d")[:5])
            raise ValueError(f"Station {code}: {remaining} missing days after processing "
                             f"(examples: {examples}). Adjust the period or check the data.")
        name = path.parent.name.replace("-", "_")
        destination = OUTPUT_DIR / f"{name}_river_stage_complete.csv"
        outputs.append((destination, series, originally_missing, filled, interpolated))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for destination, series, originally_missing, filled, interpolated in outputs:
        series.rename("river_stage").to_csv(destination, index_label="date", date_format="%Y-%m-%d")
        print(f"{destination}: {len(series)} days; originally missing: {originally_missing}; "
              f"filled from Santo Antônio: {filled}; interpolated: {interpolated}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
