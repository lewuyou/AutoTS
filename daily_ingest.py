# -*- coding: utf-8 -*-
"""Daily incremental ingestion of live time-series data into DuckDB.

Mirrors the data sources configured in ``production_example.py`` and stores the
raw output of :func:`autots.load_live_daily` — i.e. *before* the akima
interpolation, ``ffill(limit=3)``, and column-dropping clean-up that lives in
``production_example.py`` — in a long-format table so that:

  * the first run downloads the full ~6-year history,
  * every later run downloads only the tail (last cached date minus a small
    overlap buffer) and upserts it,
  * the deterministic, cheap cleaning pipeline can be re-run over the full
    history at any time without re-downloading.

Why DuckDB: single-file embedded database (no server to manage), native pandas
integration, and a primary key + ``INSERT OR REPLACE`` gives cheap upserts so
revised or backfilled source values overwrite stored rows instead of leaving a
gap at the seam. SQLite would also work, but DuckDB handles wide columns and
date arithmetic more comfortably.

Usage:
    python daily_ingest.py                  # incremental (or full on first run)
    python daily_ingest.py --full           # force a full re-download
    python daily_ingest.py --db /path/x.duckdb

Optional API keys are read from environment variables; sources without a key
are skipped exactly as in ``production_example.py`` (where those keys are None):
    FRED_API_KEY, EIA_API_KEY, NOAA_CDO_TOKEN, GSA_KEY
"""

import argparse
import datetime
import os
from typing import Optional

import pandas as pd

try:
    import duckdb
except ImportError:  # duckdb is optional until a storage function is called
    duckdb = None

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_daily.duckdb")
TABLE_NAME = "daily"
# Overlap kept when re-fetching so revised/backfilled source values overwrite
# recently stored rows instead of leaving a gap at the seam.
INCREMENTAL_BUFFER_DAYS = 30

# --- Source configuration, mirrored from production_example.py ----------------
# 多数据源下载已停用（改用 baidu_index_ingest.py 抓取百度指数入库），以下配置全部注释保留：
# FRED_SERIES = [
#     "DGS10", "T5YIE", "SP500", "DCOILWTICO", "DEXUSEU",
#     "BAMLH0A0HYM2", "DAAA", "DEXUSUK", "T10Y2Y", "DHHNGSP",
# ]
# TICKERS = ["MSFT", "PG", "YUM", "MMM", "UPS", "HON"]
# TRENDS_LIST = ["forecasting", "msft", "p&g"]
# WIKIPEDIA_PAGES = ["all", "Microsoft", "Procter_%26_Gamble", "YouTube", "United_States"]
# WEATHER_STATIONS = ["USW00014771"]
# WEATHER_YEARS = 3
# EIA_RESPONDENTS = ["MISO", "PJM", "TVA", "US48"]


def _require_duckdb():
    if duckdb is None:
        raise ImportError(
            "duckdb is required for the live data cache. Install with: "
            "python3 -m pip install --user duckdb"
        )


def _init(con):
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            series_id TEXT,
            datetime DATE,
            value DOUBLE,
            PRIMARY KEY (series_id, datetime)
        )
        """
    )


def _connect(db_path=DEFAULT_DB_PATH):
    _require_duckdb()
    con = duckdb.connect(db_path)
    _init(con)
    return con


def last_cached_date(db_path=DEFAULT_DB_PATH) -> Optional[pd.Timestamp]:
    """Most recent date already cached, or None when the cache is empty."""
    con = _connect(db_path)
    try:
        out = con.execute(f"SELECT MAX(datetime) FROM {TABLE_NAME}").fetchone()[0]
    finally:
        con.close()
    return None if out is None else pd.Timestamp(out)


def save_long(df, db_path=DEFAULT_DB_PATH) -> int:
    """Upsert a long dataframe with columns ``series_id``, ``datetime``, ``value``."""
    required = ["series_id", "datetime", "value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"long data is missing columns: {missing}")
    keep = df[required].copy()
    keep["datetime"] = pd.to_datetime(keep["datetime"], errors="coerce")
    keep["value"] = pd.to_numeric(keep["value"], errors="coerce")
    keep = keep.dropna(subset=["datetime", "value"])
    if keep.empty:
        return 0
    keep = keep.drop_duplicates(subset=["series_id", "datetime"], keep="last")
    con = _connect(db_path)
    try:
        con.register("incoming", keep)
        con.execute(
            f"""
            INSERT OR REPLACE INTO {TABLE_NAME} (series_id, datetime, value)
            SELECT series_id, CAST(datetime AS DATE) AS datetime, value
            FROM incoming
            """
        )
    finally:
        con.close()
    return len(keep)


def save_wide(df, db_path=DEFAULT_DB_PATH) -> int:
    """Melt a wide dataframe (index = datetime, columns = series) and upsert it."""
    if df is None or df.empty:
        return 0
    index_name = df.index.name or "datetime"
    long = df.reset_index().melt(
        id_vars=[index_name], var_name="series_id", value_name="value"
    )
    return save_long(long.rename(columns={index_name: "datetime"}), db_path=db_path)


def load_wide(db_path=DEFAULT_DB_PATH, observation_start=None, observation_end=None) -> pd.DataFrame:
    """Read the cache and pivot back to a wide dataframe (index = datetime)."""
    con = _connect(db_path)
    query = f"SELECT series_id, datetime, value FROM {TABLE_NAME}"
    conditions = []
    params = []
    if observation_start is not None:
        conditions.append("datetime >= ?")
        params.append(pd.Timestamp(observation_start).strftime("%Y-%m-%d"))
    if observation_end is not None:
        conditions.append("datetime <= ?")
        params.append(pd.Timestamp(observation_end).strftime("%Y-%m-%d"))
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    try:
        long = con.execute(query, params).df()
    finally:
        con.close()
    if long.empty:
        return pd.DataFrame()
    wide = long.pivot(index="datetime", columns="series_id", values="value")
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "datetime"
    wide.columns.name = None
    return wide.sort_index()


def incremental_start(db_path=DEFAULT_DB_PATH, buffer_days=INCREMENTAL_BUFFER_DAYS) -> Optional[str]:
    """``observation_start`` value to pass to ``load_live_daily``.

    Returns None when the cache is empty (first run: fetch the full window);
    otherwise returns the last cached date minus ``buffer_days`` so only the
    tail is downloaded and the overlap is upserted over.
    """
    last = last_cached_date(db_path=db_path)
    if last is None:
        return None
    return (last - datetime.timedelta(days=buffer_days)).strftime("%Y-%m-%d")


# 多数据源下载已停用，以下 _source_kwargs / ingest_daily / main 全部注释保留：
# def _source_kwargs():
#     """load_live_daily arguments, matching production_example.py's call.
#
#     Keys default to None (so those sources are skipped), but can be enabled by
#     setting the corresponding environment variable.
#     """
#     return dict(
#         long=False,
#         fred_key=os.environ.get("FRED_API_KEY"),
#         fred_series=FRED_SERIES,
#         tickers=TICKERS,
#         trends_list=TRENDS_LIST,
#         weather_stations=WEATHER_STATIONS,
#         weather_years=WEATHER_YEARS,
#         noaa_cdo_token=os.environ.get("NOAA_CDO_TOKEN"),
#         earthquake_min_magnitude=None,
#         london_air_days=700,
#         wikipedia_pages=WIKIPEDIA_PAGES,
#         gsa_key=os.environ.get("GSA_KEY"),
#         gov_domain_list=None,
#         gov_domain_limit=700,
#         weather_event_types=None,
#         caiso_query=None,
#         eia_key=os.environ.get("EIA_API_KEY"),
#         eia_respondents=EIA_RESPONDENTS,
#     )


# def ingest_daily(
#     db_path=DEFAULT_DB_PATH,
#     full=False,
#     observation_start=None,
#     sleep_seconds=15,
# ):
#     """Download (tail or full window) and upsert into the cache. Returns a summary dict."""
#     from autots import load_live_daily
#
#     before = last_cached_date(db_path=db_path)
#     if observation_start is None and not full:
#         observation_start = incremental_start(db_path=db_path)
#
#     df = load_live_daily(
#         observation_start=observation_start,
#         sleep_seconds=sleep_seconds,
#         **_source_kwargs(),
#     )
#     n_rows = save_wide(df, db_path=db_path)
#     after = last_cached_date(db_path=db_path)
#     return {
#         "cached_before": before,
#         "observation_start": observation_start,
#         "series": df.shape[1],
#         "rows_fetched": df.shape[0],
#         "rows_upserted": n_rows,
#         "cached_after": after,
#     }


# def main():
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument("--db", default=DEFAULT_DB_PATH, help="path to the DuckDB file")
#     parser.add_argument("--full", action="store_true", help="force a full re-download")
#     parser.add_argument("--observation-start", default=None, help="explicit YYYY-MM-DD start")
#     parser.add_argument("--sleep-seconds", type=float, default=15)
#     args = parser.parse_args()
#
#     summary = ingest_daily(
#         db_path=args.db,
#         full=args.full,
#         observation_start=args.observation_start,
#         sleep_seconds=args.sleep_seconds,
#     )
#     print(
#         "Ingest complete: "
#         f"cached_before={summary['cached_before']}, "
#         f"observation_start={summary['observation_start']}, "
#         f"series={summary['series']}, "
#         f"rows_fetched={summary['rows_fetched']}, "
#         f"rows_upserted={summary['rows_upserted']}, "
#         f"cached_after={summary['cached_after']}"
#     )


# if __name__ == "__main__":
#     main()
