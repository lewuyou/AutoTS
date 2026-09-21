# -*- coding: utf-8 -*-
"""东方财富业绩报告抓取入库模块（每股收益/营业收入/归母净利润，季频）。

可独立运行，也可由 download_all.py 调用 run()。
https://emdata.eastmoney.com/nbjb/detail.html?fc=300999&fn=%E9%87%91%E9%BE%99%E9%B1%BC
用法（独立运行）：
    python -m AutoTS.download.nbjb                          # 抓默认股票
    python -m AutoTS.download.nbjb --codes 300999,601318    # 指定股票代码

接口（无需 Cookie / token，GET 即可）：
    RPT_LICO_FN_CPD_BB: 业绩报告，按报告期倒序，单页可返回全部
    字段: BASIC_EPS 每股收益(元), TOTAL_OPERATE_INCOME 营业总收入(元) + _TQ 同比%,
          PARENT_NETPROFIT 归母净利润(元) + _TQ 同比%, REPORTDATE 报告期, NOTICE_DATE 公告日

入库表 nbjb 字段含义（季频，主键 (code, report_date)）：
    code                      股票代码（6 位数字）
    name                      股票简称
    report_date               报告期截止日（如 2021-03-31 表示 2021 年一季报期末）
    report_q                  报告期季度标识（如 "2021Q1"，来自 REPORTDATEWZ）
    report_label              报告期中文标签（如 "2021年 一季报"，来自 REPORTDATEYW）
    eps                       基本每股收益（元，来自 BASIC_EPS）
    total_operate_income      营业总收入（元，当季累计值）
    total_operate_income_yoy  营业总收入同比（%，来自 TOTAL_OPERATE_INCOME_TQ）
    parent_netprofit          归母净利润（元，当季累计值）
    parent_netprofit_yoy      归母净利润同比（%，来自 PARENT_NETPROFIT_TQ）
    notice_date               公告日期（实际披露日）

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
DEFAULT_CODES = ["300999"]
TABLE_NAME = "nbjb"

API_BASE = "https://datacenter.eastmoney.com/securities/api/data/get"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

STY = ",".join([
    "SECURITY_CODE", "SECURITY_NAME_ABBR", "REPORTDATE", "REPORTDATEWZ", "REPORTDATEYW",
    "BASIC_EPS", "TOTAL_OPERATE_INCOME", "TOTAL_OPERATE_INCOME_TQ",
    "PARENT_NETPROFIT", "PARENT_NETPROFIT_TQ", "NOTICE_DATE",
])


def _require_duckdb():
    if duckdb is None:
        raise ImportError("缺少 duckdb，请安装：python3 -m pip install duckdb")


def _to_date(s):
    """'2026-06-30 00:00:00' -> '2026-06-30'"""
    return (s or "").split(" ")[0]


def fetch_stock(code):
    """抓单只股票业绩报告，返回行列表。"""
    url = (
        f"{API_BASE}?type=RPT_LICO_FN_CPD_BB&source=DataCenter&client=WAP"
        f"&sty={STY}&p=1&ps=200&sr=-1&st=REPORTDATE"
        f"&filter=(SECURITY_CODE=%22{code}%22)"
    )
    try:
        r = subprocess.run(
            ["curl", "-s", "-f", url, "-H", f"User-Agent: {UA}", "-H", "Accept: application/json"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            print(f"  [业绩] {code} curl exit {r.returncode}", file=sys.stderr)
            return []
        j = json.loads(r.stdout)
        data = (j.get("result") or {}).get("data") or []
    except Exception as e:
        print(f"  [业绩] {code} 异常: {e}", file=sys.stderr)
        return []

    rows = []
    for rec in data:
        rows.append((
            code,
            rec.get("SECURITY_NAME_ABBR"),
            _to_date(rec.get("REPORTDATE")),
            rec.get("REPORTDATEWZ"),
            rec.get("REPORTDATEYW"),
            rec.get("BASIC_EPS"),
            rec.get("TOTAL_OPERATE_INCOME"),
            rec.get("TOTAL_OPERATE_INCOME_TQ"),
            rec.get("PARENT_NETPROFIT"),
            rec.get("PARENT_NETPROFIT_TQ"),
            _to_date(rec.get("NOTICE_DATE")),
        ))
    rows.sort(key=lambda r: (r[0], r[2]))
    return rows


def ingest(rows, db_path=DEFAULT_DB_PATH):
    """UPSERT 到 DuckDB 的 nbjb 表。"""
    _require_duckdb()
    con = duckdb.connect(db_path)
    try:
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                code TEXT,
                name TEXT,
                report_date DATE,
                report_q TEXT,
                report_label TEXT,
                eps DOUBLE,
                total_operate_income DOUBLE,
                total_operate_income_yoy DOUBLE,
                parent_netprofit DOUBLE,
                parent_netprofit_yoy DOUBLE,
                notice_date DATE,
                PRIMARY KEY (code, report_date)
            )
            """
        )
        con.executemany(
            f"""INSERT OR REPLACE INTO {TABLE_NAME}
                (code, name, report_date, report_q, report_label,
                 eps, total_operate_income, total_operate_income_yoy,
                 parent_netprofit, parent_netprofit_yoy, notice_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    finally:
        con.close()
    return len(rows)


def run(codes=None, db_path=None):
    """供外部调用的入口。

    参数：
        codes: 股票代码列表（6 位数字，如 ["300999"]），默认 ["300999"]
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
        print(f"[业绩] 抓取 {code} ...")
        rows = fetch_stock(code)
        print(f"[业绩] {code} 抓到 {len(rows)} 行")
        all_rows.extend(rows)

    db_path = db_path or DEFAULT_DB_PATH
    n = ingest(all_rows, db_path=db_path)
    print(f"[业绩] 入库 {n} 行 -> {db_path}")

    series = {}
    for code, name, rd, *_ in all_rows:
        key = f"{code}_{name}"
        if key not in series:
            series[key] = {"n": 0, "first": rd, "last": rd}
        series[key]["n"] += 1
        series[key]["last"] = rd
        if series[key]["n"] == 1:
            series[key]["first"] = rd

    print("\n[业绩] 入库概览")
    for key in sorted(series):
        s = series[key]
        print(f"  {key}: n={s['n']:3d}  {s['first']} ~ {s['last']}")

    return {"rows_count": n, "db_path": db_path, "series": series}


def main():
    parser = argparse.ArgumentParser(description="东方财富业绩报告抓取入库（EPS/营收/净利润）")
    parser.add_argument("--codes", default=",".join(DEFAULT_CODES), help="逗号分隔股票代码，如 300999,601318")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB 文件路径")
    args = parser.parse_args()
    run(codes=args.codes, db_path=args.db)


if __name__ == "__main__":
    main()
