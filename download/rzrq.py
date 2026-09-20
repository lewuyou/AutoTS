# -*- coding: utf-8 -*-
"""东方财富个股融资融券抓取入库模块（日频）。

可独立运行，也可由 download_all.py 调用 run()。

用法（独立运行）：
    python -m AutoTS.download.rzrq                          # 抓默认股票
    python -m AutoTS.download.rzrq --codes 688223,300999    # 指定股票代码

接口（无需 Cookie / token，GET 即可）：
    RPT_MARGIN_STATISTICS_STOCKS，按 TRADE_DATE 倒序分页，ps 最大 500
    filter=(SECURITY_CODE="688223")

入库表 rzrq 字段含义（金额单位：元；量单位：股；比率为 %）：
    code                 股票代码（6 位数字）
    name                 股票简称
    date                 交易日期
    margin_balance       两融余额（融资余额+融券余额，元）
    margin_balance_ratio 两融余额占流通市值比（%）
    fin_balance          融资余额（元）
    fin_balance_ratio    融资余额占流通市值比（%）
    loan_balance         融券余额（元）
    loan_balance_ratio   融券余额占流通市值比（%）
    fin_netbuy_amt       融资净买入额（融资买入-融资偿还，元）
    fin_tval_ratio       融资净买入额占成交额比（%）
    fin_buy_amt          融资买入额（元）
    fin_repay_amt        融资偿还额（元）
    loan_balance_vol     融券余量（股）
    loan_netsell_amt     融券净卖出额（元）
    loan_tval_ratio      融券净卖出额占成交额比（%）
    loan_netsell_vol     融券净卖出量（股）
    loan_sell_vol        融券卖出量（股）
    loan_repay_vol       融券偿还量（股）

依赖：
    pip install duckdb
"""

import argparse
import json
import os
import subprocess
import sys

try:
    import duckdb
except ImportError:
    duckdb = None

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autots.duckdb")
DEFAULT_CODES = ["688223"]
TABLE_NAME = "rzrq"
PAGE_SIZE = 500

API_BASE = "https://datacenter.eastmoney.com/securities/api/data/get"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

COLUMNS = [
    "SECUCODE", "SECURITY_CODE", "TRADE_DATE", "SECURITY_NAME_ABBR",
    "MARGIN_BALANCE", "MARGIN_BALANCE_RATIO",
    "FIN_BALANCE", "FIN_BALANCE_RATIO", "LOAN_BALANCE", "LOAN_BALANCE_RATIO",
    "FIN_NETBUY_AMT", "FIN_TVAL_RATIO", "FIN_BUY_AMT", "FIN_REPAY_AMT",
    "LOAN_BALANCE_VOL", "LOAN_NETSELL_AMT", "LOAN_TVAL_RATIO",
    "LOAN_NETSELL_VOL", "LOAN_SELL_VOL", "LOAN_REPAY_VOL",
]

# 与 COLUMNS 顺序对应的入库字段（去掉 SECUCODE/SECURITY_CODE/TRADE_DATE/SECURITY_NAME_ABBR 表头）
FIELD_KEYS = [
    "MARGIN_BALANCE", "MARGIN_BALANCE_RATIO",
    "FIN_BALANCE", "FIN_BALANCE_RATIO", "LOAN_BALANCE", "LOAN_BALANCE_RATIO",
    "FIN_NETBUY_AMT", "FIN_TVAL_RATIO", "FIN_BUY_AMT", "FIN_REPAY_AMT",
    "LOAN_BALANCE_VOL", "LOAN_NETSELL_AMT", "LOAN_TVAL_RATIO",
    "LOAN_NETSELL_VOL", "LOAN_SELL_VOL", "LOAN_REPAY_VOL",
]


def _require_duckdb():
    if duckdb is None:
        raise ImportError("缺少 duckdb，请安装：python3 -m pip install duckdb")


def _to_date(s):
    """'2026-09-18 00:00:00' -> '2026-09-18'"""
    return (s or "").split(" ")[0]


def _fetch_page(code, page):
    """抓单页，返回 (data列表, 总页数)。失败返回 ([], 0)。"""
    url = (
        f"{API_BASE}?type=RPT_MARGIN_STATISTICS_STOCKS&sty={','.join(COLUMNS)}"
        f"&p={page}&ps={PAGE_SIZE}&sr=-1&st=TRADE_DATE&source=DataCenter&client=WAP"
        f"&filter=(SECURITY_CODE=%22{code}%22)"
    )
    try:
        r = subprocess.run(
            ["curl", "-s", "-f", url, "-H", f"User-Agent: {UA}", "-H", "Accept: application/json"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            print(f"  [两融] {code} p{page} curl exit {r.returncode}", file=sys.stderr)
            return [], 0
        j = json.loads(r.stdout)
        res = j.get("result") or {}
        return res.get("data") or [], res.get("pages") or 0
    except Exception as e:
        print(f"  [两融] {code} p{page} 异常: {e}", file=sys.stderr)
        return [], 0


def fetch_stock(code):
    """抓单只股票全部两融数据（自动翻页），返回行列表。"""
    rows = []
    page = 1
    pages = 1
    while page <= pages:
        data, pages = _fetch_page(code, page)
        if not data:
            break
        for rec in data:
            rows.append(tuple(
                [code, rec.get("SECURITY_NAME_ABBR"), _to_date(rec.get("TRADE_DATE"))]
                + [rec.get(k) for k in FIELD_KEYS]
            ))
        page += 1
    rows.sort(key=lambda r: (r[0], r[2]))
    return rows


def ingest(rows, db_path=DEFAULT_DB_PATH):
    """UPSERT 到 DuckDB 的 rzrq 表。"""
    _require_duckdb()
    cols = ", ".join([
        "code", "name", "date",
        "margin_balance", "margin_balance_ratio",
        "fin_balance", "fin_balance_ratio", "loan_balance", "loan_balance_ratio",
        "fin_netbuy_amt", "fin_tval_ratio", "fin_buy_amt", "fin_repay_amt",
        "loan_balance_vol", "loan_netsell_amt", "loan_tval_ratio",
        "loan_netsell_vol", "loan_sell_vol", "loan_repay_vol",
    ])
    con = duckdb.connect(db_path)
    try:
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                code TEXT,
                name TEXT,
                date DATE,
                margin_balance DOUBLE,
                margin_balance_ratio DOUBLE,
                fin_balance DOUBLE,
                fin_balance_ratio DOUBLE,
                loan_balance DOUBLE,
                loan_balance_ratio DOUBLE,
                fin_netbuy_amt DOUBLE,
                fin_tval_ratio DOUBLE,
                fin_buy_amt DOUBLE,
                fin_repay_amt DOUBLE,
                loan_balance_vol DOUBLE,
                loan_netsell_amt DOUBLE,
                loan_tval_ratio DOUBLE,
                loan_netsell_vol DOUBLE,
                loan_sell_vol DOUBLE,
                loan_repay_vol DOUBLE,
                PRIMARY KEY (code, date)
            )
            """
        )
        placeholders = ", ".join(["?"] * (3 + len(FIELD_KEYS)))
        con.executemany(
            f"INSERT OR REPLACE INTO {TABLE_NAME} ({cols}) VALUES ({placeholders})",
            rows,
        )
    finally:
        con.close()
    return len(rows)


def run(codes=None, db_path=None):
    """供外部调用的入口。

    参数：
        codes: 股票代码列表（6 位数字，如 ["688223"]），默认 ["688223"]
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

    all_rows = []
    for code in codes:
        print(f"[两融] 抓取 {code} ...")
        rows = fetch_stock(code)
        print(f"[两融] {code} 抓到 {len(rows)} 行")
        all_rows.extend(rows)

    db_path = db_path or DEFAULT_DB_PATH
    n = ingest(all_rows, db_path=db_path)
    print(f"[两融] 入库 {n} 行 -> {db_path}")

    series = {}
    for code, name, d, *_ in all_rows:
        key = f"{code}_{name}"
        if key not in series:
            series[key] = {"n": 0, "first": d, "last": d}
        series[key]["n"] += 1
        series[key]["last"] = d
        if series[key]["n"] == 1:
            series[key]["first"] = d

    print("\n[两融] 入库概览")
    for key in sorted(series):
        s = series[key]
        print(f"  {key}: n={s['n']:5d}  {s['first']} ~ {s['last']}")

    return {"rows_count": n, "db_path": db_path, "series": series}


def main():
    parser = argparse.ArgumentParser(description="东方财富个股融资融券抓取入库（日频）")
    parser.add_argument("--codes", default=",".join(DEFAULT_CODES), help="逗号分隔股票代码，如 688223,300999")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB 文件路径")
    args = parser.parse_args()
    run(codes=args.codes, db_path=args.db)


if __name__ == "__main__":
    main()
