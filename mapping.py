#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
PLEIADES_DIR = BASE_DIR / "pleiades"

# Samian file is fixed now (root folder)
SAMIAN_PATH = BASE_DIR / "samianresearch.csv"

FILES = {
    "places": "places.csv",
    "names": "names.csv",
    "loc_points": "location_points.csv",
    "places_accuracy": "places_accuracy.csv",
    "place_types": "place_types.csv",
    "time_periods": "time_periods.csv",
    "loc_polygons": "location_polygons.csv",
}

ERA_LOCATION_FILTER = {
    "association_certainty": {"certain", "probable"},
    "location_precision": {"precise"},
}

ALLOW_ROUGH_LOCATIONS_FOR_ERA = False
MAP_TO_TIME_PERIODS = True


# -----------------------------
# Helpers
# -----------------------------
def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(
        path,
        sep=",",
        encoding="utf-8-sig",
        dtype=str,  # load as str first, then convert
        keep_default_na=False,  # keep empty as ""
        low_memory=False,
    )
    df.columns = df.columns.str.strip()
    return df


_BC_RE = re.compile(r"^\s*(\d+)\s*BC\s*$", re.IGNORECASE)
_AD_RE = re.compile(r"^\s*(\d+)\s*(AD|CE)?\s*$", re.IGNORECASE)


def parse_time_bound(value: str) -> Optional[int]:
    v = (value or "").strip()
    if not v:
        return None

    m = _BC_RE.match(v)
    if m:
        return -int(m.group(1))

    m = _AD_RE.match(v)
    if m:
        return int(m.group(1))

    try:
        return int(float(v))
    except ValueError:
        return None


def to_float(value: str) -> Optional[float]:
    v = (value or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def interval_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    a0, a1 = a
    b0, b1 = b
    left = max(a0, b0)
    right = min(a1, b1)
    return max(0, right - left)


def compute_overlapping_time_periods(
    start, end, periods_df: pd.DataFrame
) -> Tuple[str, str]:
    if start is None or end is None:
        return "", ""
    if start > end:
        start, end = end, start

    overlaps = []
    for _, r in periods_df.iterrows():
        lb = r["lower_i"]
        ub = r["upper_i"]
        if lb is None or ub is None:
            continue
        p0, p1 = min(int(lb), int(ub)), max(int(lb), int(ub))
        if interval_overlap((start, end), (p0, p1)) > 0:
            overlaps.append((r["key"], r["term"]))

    if not overlaps:
        return "", ""
    return (
        "|".join([k for k, _ in overlaps]),
        "|".join([t for _, t in overlaps]),
    )


# -----------------------------
# Pleiades builders (unchanged core)
# -----------------------------
def build_places_core(places_df: pd.DataFrame) -> pd.DataFrame:
    core = places_df.copy()
    core["pleiades_id"] = core["id"].astype(str).str.strip()
    core["pleiades_uri"] = core["uri"].astype(str).str.strip()
    core["label"] = core["title"].astype(str).str.strip()
    core["latitude"] = core["representative_latitude"].apply(to_float)
    core["longitude"] = core["representative_longitude"].apply(to_float)
    core["bounding_box_wkt"] = core.get("bounding_box_wkt", "").astype(str)
    core["location_precision_place"] = (
        core.get("location_precision", "").astype(str).str.strip()
    )
    return core[
        [
            "pleiades_id",
            "pleiades_uri",
            "label",
            "latitude",
            "longitude",
            "bounding_box_wkt",
            "location_precision_place",
        ]
    ]


def build_alt_labels(names_df: pd.DataFrame) -> pd.DataFrame:
    df = names_df.copy()
    df["place_id"] = df["place_id"].astype(str).str.strip()

    cols = [
        c
        for c in [
            "attested_form",
            "romanized_form_1",
            "romanized_form_2",
            "romanized_form_3",
            "title",
        ]
        if c in df.columns
    ]
    parts = []
    for c in cols:
        parts.append(df[["place_id", c]].rename(columns={c: "name"}))

    if not parts:
        return pd.DataFrame(columns=["pleiades_id", "alt_labels"])

    long = pd.concat(parts, ignore_index=True)
    long["name"] = long["name"].astype(str).str.strip()
    long = long[long["name"] != ""]

    if "association_certainty" in df.columns:
        cert = df[["place_id", "association_certainty"]].copy()
        cert["association_certainty"] = (
            cert["association_certainty"].astype(str).str.strip().str.lower()
        )
        long = long.merge(cert, on="place_id", how="left")
        long = long[long["association_certainty"].isin({"certain", "probable", ""})]
        long = long.drop(columns=["association_certainty"], errors="ignore")

    long = long.drop_duplicates(subset=["place_id", "name"])

    return (
        long.groupby("place_id")["name"]
        .apply(lambda s: "|".join(sorted(set(s.tolist()))))
        .reset_index()
        .rename(columns={"place_id": "pleiades_id", "name": "alt_labels"})
    )


def build_place_types(placetypes_df: pd.DataFrame) -> pd.DataFrame:
    df = placetypes_df.copy()
    df["place_id"] = df["place_id"].astype(str).str.strip()
    df["place_type"] = df["place_type"].astype(str).str.strip()
    df = df[df["place_type"] != ""].drop_duplicates()
    return (
        df.groupby("place_id")["place_type"]
        .apply(lambda s: "|".join(sorted(set(s.tolist()))))
        .reset_index()
        .rename(columns={"place_id": "pleiades_id", "place_type": "place_types"})
    )


def build_accuracy(accuracy_df: pd.DataFrame) -> pd.DataFrame:
    df = accuracy_df.copy()
    df["place_id"] = df["place_id"].astype(str).str.strip()

    def fnum(x):
        try:
            v = float(str(x).strip())
            return None if v < 0 else v
        except Exception:
            return None

    df["max_accuracy_m"] = df.get("max_accuracy_meters", "").apply(fnum)
    df["min_accuracy_m"] = df.get("min_accuracy_meters", "").apply(fnum)
    df["accuracy_hull_wkt"] = df.get("accuracy_hull", "").astype(str)
    df["location_precision_accuracy"] = (
        df.get("location_precision", "").astype(str).str.strip()
    )

    out = df.rename(columns={"place_id": "pleiades_id"})
    return out[
        [
            "pleiades_id",
            "location_precision_accuracy",
            "accuracy_hull_wkt",
            "min_accuracy_m",
            "max_accuracy_m",
        ]
    ].drop_duplicates(subset=["pleiades_id"])


def build_era_from_locations(loc_points_df: pd.DataFrame) -> pd.DataFrame:
    df = loc_points_df.copy()
    df["place_id"] = df["place_id"].astype(str).str.strip()
    df["association_certainty"] = (
        df.get("association_certainty", "").astype(str).str.strip().str.lower()
    )
    df["location_precision"] = (
        df.get("location_precision", "").astype(str).str.strip().str.lower()
    )

    cert_ok = df["association_certainty"].isin(
        ERA_LOCATION_FILTER["association_certainty"]
    )
    if ALLOW_ROUGH_LOCATIONS_FOR_ERA:
        prec_ok = df["location_precision"].isin({"precise", "rough"})
    else:
        prec_ok = df["location_precision"].isin(
            ERA_LOCATION_FILTER["location_precision"]
        )

    df = df[cert_ok & prec_ok].copy()
    df["tpq"] = df.get("year_after_which", "").apply(parse_time_bound)
    df["taq"] = df.get("year_before_which", "").apply(parse_time_bound)
    df = df[(df["tpq"].notna()) | (df["taq"].notna())].copy()

    if df.empty:
        return pd.DataFrame(
            columns=["pleiades_id", "earliest_year", "latest_year", "era_source"]
        )

    def agg_min(vals):
        vals = [v for v in vals if v is not None]
        return int(min(vals)) if vals else None

    def agg_max(vals):
        vals = [v for v in vals if v is not None]
        return int(max(vals)) if vals else None

    grouped = (
        df.groupby("place_id")
        .agg(
            earliest_year=("tpq", lambda s: agg_min(s.tolist())),
            latest_year=("taq", lambda s: agg_max(s.tolist())),
        )
        .reset_index()
        .rename(columns={"place_id": "pleiades_id"})
    )
    grouped["era_source"] = "location_points"
    return grouped


def build_time_periods_vocab(time_periods_df: pd.DataFrame) -> pd.DataFrame:
    df = time_periods_df.copy()
    df["key"] = df["key"].astype(str).str.strip()
    df["term"] = df["term"].astype(str).str.strip()
    df["lower_i"] = df["lower_bound"].apply(parse_time_bound)
    df["upper_i"] = df["upper_bound"].apply(parse_time_bound)
    df = df[df["lower_i"].notna() & df["upper_i"].notna()].copy()
    df["lower_i"] = df["lower_i"].astype(int)
    df["upper_i"] = df["upper_i"].astype(int)
    return df[["key", "term", "lower_i", "upper_i", "same_as"]]


def build_pleiades_view() -> pd.DataFrame:
    places_df = read_csv(PLEIADES_DIR / FILES["places"])
    names_df = read_csv(PLEIADES_DIR / FILES["names"])
    loc_points_df = read_csv(PLEIADES_DIR / FILES["loc_points"])
    acc_df = read_csv(PLEIADES_DIR / FILES["places_accuracy"])
    placetypes_df = read_csv(PLEIADES_DIR / FILES["place_types"])
    time_periods_df = read_csv(PLEIADES_DIR / FILES["time_periods"])

    places_core = build_places_core(places_df)
    alt_labels = build_alt_labels(names_df)
    place_types = build_place_types(placetypes_df)
    accuracy = build_accuracy(acc_df)
    era = build_era_from_locations(loc_points_df)
    periods_vocab = build_time_periods_vocab(time_periods_df)

    df = places_core.copy()
    df = df.merge(alt_labels, on="pleiades_id", how="left")
    df = df.merge(place_types, on="pleiades_id", how="left")
    df = df.merge(accuracy, on="pleiades_id", how="left")
    df = df.merge(era, on="pleiades_id", how="left")

    df["alt_labels"] = df["alt_labels"].fillna("")
    df["place_types"] = df["place_types"].fillna("")

    df["location_precision"] = df["location_precision_accuracy"].fillna("")
    df.loc[df["location_precision"] == "", "location_precision"] = df[
        "location_precision_place"
    ].fillna("")

    keys, terms = [], []
    for _, row in df.iterrows():
        k, t = compute_overlapping_time_periods(
            row.get("earliest_year"), row.get("latest_year"), periods_vocab
        )
        keys.append(k)
        terms.append(t)
    df["time_period_keys"] = keys
    df["time_period_terms"] = terms

    return df[
        [
            "pleiades_id",
            "pleiades_uri",
            "label",
            "alt_labels",
            "latitude",
            "longitude",
            "bounding_box_wkt",
            "location_precision",
            "min_accuracy_m",
            "max_accuracy_m",
            "accuracy_hull_wkt",
            "place_types",
            "earliest_year",
            "latest_year",
            "era_source",
            "time_period_keys",
            "time_period_terms",
        ]
    ].copy()


# -----------------------------
# Samian loader (YOUR schema)
# -----------------------------
SAMIAN_REQUIRED = [
    "id",
    "label",
    "altlabels",
    "lon",
    "lat",
    "earliest_year",
    "latest_year",
    "q_start",
    "q_end",
    "q_interval",
    "unc_start_years",
    "unc_end_years",
    "unc_interval_years",
]


def load_samian_csv(path: Path) -> pd.DataFrame:
    df = read_csv(path)

    # Validate required columns (case-sensitive as in your CSV)
    missing = [c for c in SAMIAN_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Samian CSV missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    out = df.copy()

    # Normalize core fields to our internal names
    out["samian_id"] = out["id"].astype(str).str.strip()
    out["label"] = out["label"].astype(str).str.strip()
    out["alt_labels"] = out["altlabels"].astype(str).fillna("").str.strip()

    out["latitude"] = out["lat"].apply(to_float)
    out["longitude"] = out["lon"].apply(to_float)

    out["earliest_year"] = out["earliest_year"].apply(parse_time_bound)
    out["latest_year"] = out["latest_year"].apply(parse_time_bound)

    # Quality scores [0..1]
    for c in ["q_start", "q_end", "q_interval"]:
        out[c] = out[c].apply(to_float)

    # Uncertainty in years (ints)
    for c in ["unc_start_years", "unc_end_years", "unc_interval_years"]:
        out[c] = out[c].apply(parse_time_bound)

    # Defensive swap if earliest > latest
    mask = (
        out["earliest_year"].notna()
        & out["latest_year"].notna()
        & (out["earliest_year"] > out["latest_year"])
    )
    if mask.any():
        tmp = out.loc[mask, "earliest_year"].copy()
        out.loc[mask, "earliest_year"] = out.loc[mask, "latest_year"]
        out.loc[mask, "latest_year"] = tmp

    # Convenience flags
    out["has_coords"] = out["latitude"].notna() & out["longitude"].notna()
    out["has_time"] = out["earliest_year"].notna() & out["latest_year"].notna()

    # Keep a clean downstream-ready selection
    return out[
        [
            "samian_id",
            "label",
            "alt_labels",
            "latitude",
            "longitude",
            "earliest_year",
            "latest_year",
            "q_start",
            "q_end",
            "q_interval",
            "unc_start_years",
            "unc_end_years",
            "unc_interval_years",
            "has_coords",
            "has_time",
        ]
    ].copy()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print("Building Pleiades view...")
    pleiades_view_df = build_pleiades_view()
    print(f"  pleiades_view_df: {len(pleiades_view_df):,} rows")

    print("\nLoading Samian research CSV...")
    if not SAMIAN_PATH.exists():
        raise FileNotFoundError(
            f"Samian file not found: {SAMIAN_PATH}\n"
            "Please place 'samianresearch.csv' in the same folder as this script."
        )

    samian_df = load_samian_csv(SAMIAN_PATH)
    print(f"  samian_df: {len(samian_df):,} rows")
    print("\nSamian sample:")
    print(samian_df.head(10).to_string(index=False))

    print("\nPleiades sample:")
    print(pleiades_view_df.head(5).to_string(index=False))

    # Expose for interactive use (Debug Console)
    globals()["pleiades_view_df"] = pleiades_view_df
    globals()["samian_df"] = samian_df


if __name__ == "__main__":
    main()
