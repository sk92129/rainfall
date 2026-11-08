#!/usr/bin/env python3
"""Train a linear model of Miami annual rainfall and surface runoff.

Rainfall comes from NOAA's Global Summary of the Year for Miami
International Airport (GHCN station USW00012839), the official long
record for the city. Complete calendar years run from 1948 through the
last finished year.

NOAA does not observe runoff at this station. Annual surface-runoff
depths for 1996–2010 are the directly connected impervious-area
estimates published in USGS Scientific Investigations Report 2014-5162,
figure 7 (inches over the onshore Miami-Dade study area). A linear
regression maps airport rainfall onto that runoff. A second linear
regression maps year onto rainfall. A future year is predicted by
chaining the two fits.

Usage
-----
  python rainfall.py                  # download, train, forecast 10 years
  python rainfall.py 2040             # forecast one year
  python rainfall.py 2030-2040        # forecast an inclusive range
  python rainfall.py 2030 2035 2045   # forecast listed years
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.linear_model import LinearRegression

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

STATION_ID = "USW00012839"
STATION_NAME = "MIAMI INTERNATIONAL AIRPORT, FL US"
NOAA_URL = (
    "https://www.ncei.noaa.gov/access/services/data/v1"
    "?dataset=global-summary-of-the-year"
    f"&stations={STATION_ID}"
    "&startDate=1800-01-01"
    "&endDate=2100-12-31"
    "&dataTypes=PRCP"
    "&format=csv"
    "&units=standard"
    "&includeAttributes=true"
    "&includeStationName=true"
    "&includeStationLocation=1"
)
RAW_CSV = DATA_DIR / "noaa_miami_intl_gsoy_prcp.csv"
ANNUAL_CSV = DATA_DIR / "miami_annual.csv"
MODEL_PATH = MODEL_DIR / "miami_linear_regression.joblib"
FORECAST_CSV = OUTPUT_DIR / "forecast.csv"
METRICS_JSON = OUTPUT_DIR / "model_metrics.json"
PLOT_PATH = OUTPUT_DIR / "miami_rainfall_runoff.png"

# Annual surface runoff (inches) from directly connected impervious area,
# onshore Miami-Dade study area. Read from the year labels on figure 7 of
# Hughes and White, USGS SIR 2014-5162.
# https://pubs.usgs.gov/sir/2014/5162/
USGS_RUNOFF_IN = {
    1996: 2.11,
    1997: 3.30,
    1998: 3.00,
    1999: 3.65,
    2000: 3.66,
    2001: 3.60,
    2002: 3.21,
    2003: 3.31,
    2004: 3.04,
    2005: 4.31,
    2006: 3.22,
    2007: 3.87,
    2008: 3.47,
    2009: 3.68,
    2010: 3.88,
}

RUNOFF_SOURCE = (
    "USGS SIR 2014-5162 figure 7, directly connected impervious-area "
    "surface runoff, inches over the onshore Miami-Dade study area"
)


def fetch_noaa(path: Path = RAW_CSV) -> Path:
    """Download the full annual precipitation file from NOAA NCEI."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        NOAA_URL,
        headers={"User-Agent": "miami-rainfall-regression/1.0 (research)"},
        timeout=60,
    )
    response.raise_for_status()
    if not response.text.lstrip().startswith('"STATION"') and "STATION" not in response.text[:200]:
        raise RuntimeError(
            "NOAA did not return a Global Summary of the Year CSV. "
            f"Response began: {response.text[:180]!r}"
        )
    path.write_text(response.text)
    return path


def load_annual_rainfall(path: Path = RAW_CSV) -> pd.DataFrame:
    """Keep complete calendar years. Drop the year still in progress."""
    raw = pd.read_csv(path)
    raw["year"] = pd.to_numeric(raw["DATE"], errors="coerce")
    raw["rainfall_in"] = pd.to_numeric(raw["PRCP"], errors="coerce")
    annual = raw.dropna(subset=["year", "rainfall_in"]).copy()
    annual["year"] = annual["year"].astype(int)
    in_progress = date.today().year
    annual = annual.loc[annual["year"] < in_progress]
    annual = annual.sort_values("year")
    if annual["year"].duplicated().any():
        raise RuntimeError("NOAA file has more than one precipitation total for a year.")
    gaps = sorted(set(range(int(annual["year"].min()), int(annual["year"].max()) + 1)) - set(annual["year"]))
    if gaps:
        raise RuntimeError(f"NOAA annual series is missing years: {gaps}")
    columns = [
        "year",
        "STATION",
        "NAME",
        "LATITUDE",
        "LONGITUDE",
        "ELEVATION",
        "rainfall_in",
        "PRCP_ATTRIBUTES",
    ]
    return annual[columns].reset_index(drop=True)


def fit_models(annual: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Fit rainfall ~ year, and runoff ~ rainfall on the USGS overlap."""
    overlap = annual.loc[annual["year"].isin(USGS_RUNOFF_IN)].copy()
    overlap["runoff_in"] = overlap["year"].map(USGS_RUNOFF_IN)
    missing = sorted(set(USGS_RUNOFF_IN) - set(overlap["year"]))
    if missing:
        raise RuntimeError(
            "NOAA rainfall is missing years needed to fit runoff: " + ", ".join(map(str, missing))
        )

    rain_x = annual[["year"]].to_numpy(dtype=float)
    rain_y = annual["rainfall_in"].to_numpy(dtype=float)
    rainfall_model = LinearRegression()
    rainfall_model.fit(rain_x, rain_y)
    rain_stats = _ols_stats(rain_x.ravel(), rain_y, rainfall_model)

    runoff_x = overlap[["rainfall_in"]].to_numpy(dtype=float)
    runoff_y = overlap["runoff_in"].to_numpy(dtype=float)
    runoff_model = LinearRegression()
    runoff_model.fit(runoff_x, runoff_y)
    runoff_stats = _ols_stats(runoff_x.ravel(), runoff_y, runoff_model)

    estimated = annual.copy()
    estimated["usgs_runoff_in"] = estimated["year"].map(USGS_RUNOFF_IN)
    estimated["runoff_in"] = runoff_model.predict(estimated[["rainfall_in"]].to_numpy(dtype=float))

    bundle = {
        "rainfall_model": rainfall_model,
        "runoff_model": runoff_model,
        "rain_stats": rain_stats,
        "runoff_stats": runoff_stats,
        "station_id": STATION_ID,
        "station_name": STATION_NAME,
        "first_year": int(annual["year"].min()),
        "last_year": int(annual["year"].max()),
        "runoff_source": RUNOFF_SOURCE,
        "runoff_years": [int(y) for y in overlap["year"]],
    }
    return bundle, estimated


def _ols_stats(x: np.ndarray, y: np.ndarray, model: LinearRegression) -> dict:
    """Residual scale and leverage terms for a simple linear prediction interval."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    fitted = model.predict(x.reshape(-1, 1)).ravel()
    resid = y - fitted
    n = int(x.size)
    df = n - 2
    if df <= 0:
        raise RuntimeError("Need at least 3 observations to fit a line with an interval.")
    s = float(np.sqrt(np.sum(resid**2) / df))
    xbar = float(x.mean())
    sxx = float(np.sum((x - xbar) ** 2))
    slope = float(np.asarray(model.coef_).ravel()[0])
    se_slope = s / np.sqrt(sxx)
    t_slope = slope / se_slope
    p_value = float(2 * stats.t.sf(abs(t_slope), df))
    return {
        "n": n,
        "df": df,
        "sigma": s,
        "x_mean": xbar,
        "sxx": sxx,
        "r2": float(model.score(x.reshape(-1, 1), y)),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "intercept": float(model.intercept_),
        "slope": slope,
        "slope_se": float(se_slope),
        "p_value": p_value,
    }


def _prediction_se(fit: dict, x0: np.ndarray) -> np.ndarray:
    x0 = np.asarray(x0, dtype=float).ravel()
    leverage = 1.0 + 1.0 / fit["n"] + (x0 - fit["x_mean"]) ** 2 / fit["sxx"]
    return fit["sigma"] * np.sqrt(leverage)


def predict_years(bundle: dict, years: list[int]) -> pd.DataFrame:
    """Predict annual rainfall and runoff, with approximate 95% intervals."""
    years_arr = np.asarray(years, dtype=float).reshape(-1, 1)
    rainfall = bundle["rainfall_model"].predict(years_arr).ravel()
    runoff = bundle["runoff_model"].predict(rainfall.reshape(-1, 1)).ravel()

    rain_se = _prediction_se(bundle["rain_stats"], years_arr.ravel())
    # Runoff interval treats the predicted rainfall as an uncertain input
    # and adds the scatter of the rainfall-to-runoff fit.
    runoff_given_rain_se = _prediction_se(bundle["runoff_stats"], rainfall)
    slope = bundle["runoff_stats"]["slope"]
    runoff_se = np.sqrt(runoff_given_rain_se**2 + (slope**2) * rain_se**2)

    # The runoff sample is the smaller one, so its t critical value is the
    # more cautious choice for the combined runoff interval.
    t_rain = float(stats.t.ppf(0.975, bundle["rain_stats"]["df"]))
    t_runoff = float(stats.t.ppf(0.975, bundle["runoff_stats"]["df"]))

    frame = pd.DataFrame(
        {
            "year": [int(y) for y in years],
            "rainfall_in": rainfall,
            "rainfall_low_95": rainfall - t_rain * rain_se,
            "rainfall_high_95": rainfall + t_rain * rain_se,
            "runoff_in": runoff,
            "runoff_low_95": runoff - t_runoff * runoff_se,
            "runoff_high_95": runoff + t_runoff * runoff_se,
        }
    )
    return frame


def save_model(bundle: dict, path: Path = MODEL_PATH) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def metrics_payload(bundle: dict) -> dict:
    rain = bundle["rain_stats"]
    runoff = bundle["runoff_stats"]
    return {
        "station_id": bundle["station_id"],
        "station_name": bundle["station_name"],
        "rainfall_years": [bundle["first_year"], bundle["last_year"]],
        "rainfall_equation": (
            f"rainfall_in = {rain['intercept']:.6f} + {rain['slope']:.6f} * year"
        ),
        "rainfall_r2": rain["r2"],
        "rainfall_rmse_in": rain["rmse"],
        "rainfall_sigma_in": rain["sigma"],
        "rainfall_slope_p_value": rain["p_value"],
        "rainfall_n": rain["n"],
        "runoff_equation": (
            f"runoff_in = {runoff['intercept']:.6f} + {runoff['slope']:.6f} * rainfall_in"
        ),
        "runoff_r2": runoff["r2"],
        "runoff_rmse_in": runoff["rmse"],
        "runoff_slope_p_value": runoff["p_value"],
        "runoff_sigma_in": runoff["sigma"],
        "runoff_n": runoff["n"],
        "runoff_training_years": bundle["runoff_years"],
        "runoff_source": bundle["runoff_source"],
        "note": (
            "Rainfall is the NOAA annual total at Miami International Airport. "
            "Runoff is predicted from rainfall using a line fit to USGS "
            "Miami-Dade surface-runoff estimates for 1996-2010. The 95% "
            "ranges are prediction intervals for a new year, not confidence "
            "intervals for the mean trend."
        ),
    }


def write_outputs(annual: pd.DataFrame, bundle: dict, forecast: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    annual_out = annual.copy()
    annual_out["usgs_runoff_in"] = annual_out["year"].map(USGS_RUNOFF_IN)
    annual_out["runoff_from_rainfall_in"] = bundle["runoff_model"].predict(
        annual_out[["rainfall_in"]].to_numpy(dtype=float)
    )
    annual_out.to_csv(ANNUAL_CSV, index=False)
    forecast.to_csv(FORECAST_CSV, index=False, float_format="%.2f")
    METRICS_JSON.write_text(json.dumps(metrics_payload(bundle), indent=2) + "\n")
    _plot(annual_out, bundle, forecast, PLOT_PATH)


def _plot(annual: pd.DataFrame, bundle: dict, forecast: pd.DataFrame, path: Path) -> None:
    years = annual["year"].to_numpy(dtype=float)
    rain_line = bundle["rainfall_model"].predict(years.reshape(-1, 1)).ravel()
    future_years = forecast["year"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)

    ax = axes[0]
    ax.scatter(annual["year"], annual["rainfall_in"], s=18, color="#1f4e79", label="NOAA annual total", zorder=3)
    ax.plot(years, rain_line, color="#1f4e79", linewidth=1.5, label="Linear fit")
    ax.fill_between(
        future_years,
        forecast["rainfall_low_95"],
        forecast["rainfall_high_95"],
        color="#1f4e79",
        alpha=0.15,
        label="95% prediction interval",
    )
    ax.plot(future_years, forecast["rainfall_in"], color="#c45911", linewidth=1.8, label="Forecast")
    ax.set_ylabel("Rainfall (inches)")
    ax.set_title("Miami International Airport — annual rainfall and estimated runoff")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    observed = annual.dropna(subset=["usgs_runoff_in"])
    ax.scatter(
        observed["year"],
        observed["usgs_runoff_in"],
        s=28,
        color="#0d7377",
        label="USGS runoff, 1996–2010",
        zorder=3,
    )
    ax.plot(
        years,
        annual["runoff_from_rainfall_in"],
        color="#0d7377",
        linewidth=1.0,
        alpha=0.85,
        label="Runoff from observed rainfall",
    )
    ax.fill_between(
        future_years,
        forecast["runoff_low_95"],
        forecast["runoff_high_95"],
        color="#0d7377",
        alpha=0.15,
        label="95% prediction interval",
    )
    ax.plot(future_years, forecast["runoff_in"], color="#c45911", linewidth=1.8, label="Forecast")
    ax.set_ylabel("Runoff (inches)")
    ax.set_xlabel("Year")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def parse_year_tokens(tokens: list[str]) -> list[int]:
    years: list[int] = []
    for token in tokens:
        if "-" in token:
            left, right = token.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                raise SystemExit(f"Year range {token} is reversed.")
            if end - start > 500:
                raise SystemExit(f"Year range {token} is longer than 500 years.")
            years.extend(range(start, end + 1))
        else:
            years.append(int(token))
    # Preserve order, drop duplicates.
    seen: set[int] = set()
    unique: list[int] = []
    for year in years:
        if year not in seen:
            seen.add(year)
            unique.append(year)
    return unique


def default_horizon(last_observed_year: int, count: int = 10) -> list[int]:
    start = last_observed_year + 1
    return list(range(start, start + count))


def _format_p(p_value: float) -> str:
    if p_value < 0.001:
        return "< 0.001"
    return f"{p_value:.3f}"


def print_report(bundle: dict, forecast: pd.DataFrame) -> None:
    rain = bundle["rain_stats"]
    runoff = bundle["runoff_stats"]
    print(
        f"NOAA station {STATION_ID} ({STATION_NAME})\n"
        f"Annual rainfall: {bundle['first_year']}–{bundle['last_year']} "
        f"({rain['n']} complete years)"
    )
    print(
        f"Rainfall model:  inches = {rain['intercept']:.3f} + {rain['slope']:.4f} × year\n"
        f"                 R² = {rain['r2']:.3f}, RMSE = {rain['rmse']:.2f} in, "
        f"slope p = {_format_p(rain['p_value'])}"
    )
    print(
        f"Runoff model:    inches = {runoff['intercept']:.3f} + {runoff['slope']:.4f} × rainfall inches\n"
        f"                 fit on USGS Miami-Dade runoff, {bundle['runoff_years'][0]}–"
        f"{bundle['runoff_years'][-1]} (n = {runoff['n']})\n"
        f"                 R² = {runoff['r2']:.3f}, RMSE = {runoff['rmse']:.2f} in, "
        f"slope p = {_format_p(runoff['p_value'])}"
    )
    print(
        "\nRunoff is not a NOAA observation. It is the USGS 1996–2010 surface-runoff\n"
        "series, extended to other years through the rainfall relationship above."
    )
    if rain["r2"] < 0.25:
        print(
            "\nThe year trend explains little of the year-to-year rainfall. A forecast\n"
            "for a single future year is close to the long-term level, and the 95%\n"
            "interval is the more informative result."
        )
    if runoff["slope"] <= 0:
        print(
            "\nThe fitted runoff slope is not positive. Treat runoff forecasts with\n"
            "extra caution; wetter years did not line up with higher USGS runoff\n"
            "at this airport gage."
        )

    display = forecast.copy()
    for column in display.columns:
        if column != "year":
            display[column] = display[column].map(lambda value: f"{value:.2f}")
    print("\nForecast (inches):")
    print(display.to_string(index=False))
    print(f"\nSaved annual series  {ANNUAL_CSV.relative_to(ROOT)}")
    print(f"Saved model          {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved forecast       {FORECAST_CSV.relative_to(ROOT)}")
    print(f"Saved plot           {PLOT_PATH.relative_to(ROOT)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download NOAA annual rainfall for Miami International Airport, "
            "fit a linear regression, and predict rainfall and runoff for future years."
        )
    )
    parser.add_argument(
        "years",
        nargs="*",
        help="Years to predict, such as 2040 or 2030-2040. Defaults to the next 10 years.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Train from the CSV already in data/ instead of downloading again.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        years = parse_year_tokens(args.years)
    except ValueError:
        print("Years must be integers or an inclusive range such as 2030-2040.", file=sys.stderr)
        return 2

    if args.offline:
        if not RAW_CSV.exists():
            print(f"No local NOAA file at {RAW_CSV}. Run without --offline first.", file=sys.stderr)
            return 1
    else:
        print("Downloading NOAA Global Summary of the Year for Miami International Airport...")
        fetch_noaa(RAW_CSV)

    annual = load_annual_rainfall(RAW_CSV)
    bundle, _estimated = fit_models(annual)
    if not years:
        years = default_horizon(bundle["last_year"])
    forecast = predict_years(bundle, years)
    save_model(bundle)
    write_outputs(annual, bundle, forecast)
    print_report(bundle, forecast)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
