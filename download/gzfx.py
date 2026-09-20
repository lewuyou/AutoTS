# -*- coding: utf-8 -*-
"""东方财富估值通道抓取入库模块（市盈率/市净率/市销率/市现率，日频）。
https://emdata.eastmoney.com/gzfx/detail.html?fc=300999.SZ&fn=%E9%87%91%E9%BE%99%E9%B1%BC
可独立运行，也可由 download_all.py 调用 run()。

用法（独立运行）：
    python -m AutoTS.download.gzfx                          # 抓默认股票近1年
    python -m AutoTS.download.gzfx --codes 300999,601318    # 指定股票代码
    python -m AutoTS.download.gzfx --datetype 2             # 3年（周频）

接口（无需 Cookie / token，GET 即可）：
    估值走势 RPT_CUSTOM_DMSK_TREND: INDICATOR_VALUE 实际估值
    估值通道 RPT_CUSTOM_DMSK:       STOCK_PRICE 股价 + PASS1~5 通道带价格 + MULTIPLE1~5 倍数
    INDICATORTYPE: 1=市盈率 2=市净率 3=市销率 4=市现率
    DATETYPE:      1=近1年(日频) 2=近3年(周频) 3=近5年(周频) 4=近10年(周频)

依赖：
    pip install duckdb
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

try:
    import duckdb
except ImportError:
    duckdb = None

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autots.duckdb")
DEFAULT_CODES = ["300999"]
TABLE_NAME = "gzfx"

INDICATOR_TYPES = {1: "pe", 2: "pb", 3: "ps", 4: "pcf"}  # 市盈/市净/市销/市现
DATE_TYPES = {1: "1y", 2: "3y", 3: "5y", 4: "10y"}        # 1 为日频，其余周频

API_BASE = "https://datacenter.eastmoney.com/securities/api/data/get"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def _require_duckdb():
    if duckdb is None:
        raise ImportError("缺少 duckdb，请安装：python3 -m pip install duckdb")


def fetch_api(report_type, code, indicator_type, date_type):
    """GET datacenter 接口，返回 data 列表。失败返回 []。"""
    filt = f"(SECURITY_CODE%3D%22{code}%22)(INDICATORTYPE%3D{indicator_type})(DATETYPE%3D{date_type})"
    url = (
        f"{API_BASE}?type={report_type}&p=1&sr=-1&st=TRADE_DATE"
        f"&var=source=DataCenter&client=WAP&filter={filt}"
    )
    try:
        r = subprocess.run(
            ["curl", "-s", "-f", url, "-H", f"User-Agent: {UA}", "-H", "Accept: application/json"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            print(f"  [估值] {code} type={report_type} it={indicator_type} curl exit {r.returncode}", file=sys.stderr)
            return []
        j = json.loads(r.stdout)
        return (j.get("result") or {}).get("data") or []
    except Exception as e:
        print(f"  [估值] {code} type={report_type} it={indicator_type} 异常: {e}", file=sys.stderr)
        return []


def _to_date(s):
    """'2026-09-18 00:00:00' -> '2026-09-18'"""
    return (s or "").split(" ")[0]


def fetch_stock(code, date_type=1):
    """抓单只股票 4 个指标的走势 + 通道，返回行列表。

    行格式: (code, indicator, date, value, stock_price, pass1..pass5, mult1..mult5)
    走势行 value=INDICATOR_VALUE，通道字段为 None；通道行反之。
    这里把两类按 (code, indicator, date) 合并成一行。
    """
    merged = {}  # (indicator, date) -> dict
    for it, iname in INDICATOR_TYPES.items():
        for rec in fetch_api("RPT_CUSTOM_DMSK_TREND", code, it, date_type):
            d = _to_date(rec.get("TRADE_DATE"))
            if not d:
                continue
            m = merged.setdefault((iname, d), {})
            m["value"] = rec.get("INDICATOR_VALUE")
        for rec in fetch_api("RPT_CUSTOM_DMSK", code, it, date_type):
            d = _to_date(rec.get("TRADE_DATE"))
            if not d:
                continue
            m = merged.setdefault((iname, d), {})
            m["stock_price"] = rec.get("STOCK_PRICE")
            for i in range(1, 6):
                m[f"pass{i}"] = rec.get(f"PASS{i}")
                m[f"mult{i}"] = rec.get(f"MULTIPLE{i}")

    rows = []
    for (iname, d), m in merged.items():
        rows.append((
            code, iname, d,
            m.get("value"), m.get("stock_price"),
            m.get("pass1"), m.get("pass2"), m.get("pass3"), m.get("pass4"), m.get("pass5"),
            m.get("mult1"), m.get("mult2"), m.get("mult3"), m.get("mult4"), m.get("mult5"),
        ))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return rows


def ingest(rows, db_path=DEFAULT_DB_PATH):
    """UPSERT 到 DuckDB 的 gzfx 表。"""
    _require_duckdb()
    con = duckdb.connect(db_path)
    try:
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                code TEXT,
                indicator TEXT,
                date DATE,
                value DOUBLE,
                stock_price DOUBLE,
                pass1 DOUBLE, pass2 DOUBLE, pass3 DOUBLE, pass4 DOUBLE, pass5 DOUBLE,
                mult1 DOUBLE, mult2 DOUBLE, mult3 DOUBLE, mult4 DOUBLE, mult5 DOUBLE,
                PRIMARY KEY (code, indicator, date)
            )
            """
        )
        con.executemany(
            f"""INSERT OR REPLACE INTO {TABLE_NAME}
                (code, indicator, date, value, stock_price,
                 pass1, pass2, pass3, pass4, pass5,
                 mult1, mult2, mult3, mult4, mult5)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    finally:
        con.close()
    return len(rows)


def run(codes=None, date_type=1, db_path=None):
    """供外部调用的入口。

    参数：
        codes: 股票代码列表（6 位数字，如 ["300999"]），默认 ["300999"]
        date_type: 1=近1年(日频) 2=近3年 3=近5年 4=近10年（后三个周频）
        db_path: DuckDB 文件路径
    返回：
        dict 包含 rows_count, db_path, series 等
    """
    if codes is None:
        codes = DEFAULT_CODES
    if isinstance(codes, str):
        codes = [c.strip() for c in codes.split(",") if c.strip()]
    if not codes:
        raise ValueError("未提供股票代码")
    if date_type not in DATE_TYPES:
        raise ValueError(f"date_type 必须是 {list(DATE_TYPES)}，收到 {date_type}")

    all_rows = []
    for code in codes:
        print(f"[估值] 抓取 {code} ({DATE_TYPES[date_type]}) ...")
        rows = fetch_stock(code, date_type=date_type)
        print(f"[估值] {code} 抓到 {len(rows)} 行")
        all_rows.extend(rows)

    db_path = db_path or DEFAULT_DB_PATH
    n = ingest(all_rows, db_path=db_path)
    print(f"[估值] 入库 {n} 行 -> {db_path}")

    series = {}
    for code, iname, d, *_ in all_rows:
        key = f"{code}_{iname}"
        if key not in series:
            series[key] = {"n": 0, "first": d, "last": d}
        series[key]["n"] += 1
        series[key]["last"] = d
        if series[key]["n"] == 1:
            series[key]["first"] = d

    print("\n[估值] 入库概览")
    for key in sorted(series):
        s = series[key]
        print(f"  {key}: n={s['n']:5d}  {s['first']} ~ {s['last']}")

    return {"rows_count": n, "db_path": db_path, "series": series}


def main():
    parser = argparse.ArgumentParser(description="东方财富估值通道抓取入库（PE/PB/PS/PCF）")
    parser.add_argument("--codes", default=",".join(DEFAULT_CODES), help="逗号分隔股票代码，如 300999,601318")
    parser.add_argument("--datetype", type=int, default=1, choices=[1, 2, 3, 4],
                        help="1=近1年(日频) 2=近3年 3=近5年 4=近10年（后三个周频），默认 1")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB 文件路径")
    args = parser.parse_args()
    run(codes=args.codes, date_type=args.datetype, db_path=args.db)


if __name__ == "__main__":
    main()
