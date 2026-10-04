# -*- coding: utf-8 -*-
"""东方财富业绩报告抓取入库模块（每股收益/营业收入/归母净利润，季频）。

可独立运行，也可由 download_all.py 调用 run()。
https://emdata.eastmoney.com/nbjb/detail.html?fc=300999&fn=%E9%87%91%E9%BE%99%E9%B1%BC

用法（独立运行）：
    python -m download.nbjb                          # 从 stock_list 抓取全部 A 股业绩报告
    python -m download.nbjb --codes 300999,601318    # 指定股票代码
    python -m download.nbjb --stage auto             # 暂存模式，事后 --merge 合并入主库

模式说明：
    * 不传 --codes 时，从 DuckDB stock_list 表读取全部在市股票（自动排除名称含 ST 的）。
    * 业绩报告为季频，单页返回该股全部报告期，UPSERT 幂等，每天重跑即可刷到最新披露。
    * 失败清单写入 download/temp/nbjb_failed_<时间戳>.txt，重跑即可补抓。

暂存模式（--stage）：
    数据不入主库，改写入暂存 DuckDB（表结构与主库 nbjb 完全相同），
    供多个并行任务各自生成暂存文件后，用 --merge 统一合并入主库：
        python -m download.nbjb --stage auto --codes 000001,600000
        python -m download.nbjb --merge "download/temp/nbjb_stage_*.duckdb"
    合并成功的暂存文件自动重命名加 .merged 后缀，防止重复合并。

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
import datetime
import json
import os
import subprocess
import sys
import time

try:
    import duckdb
except ImportError:
    duckdb = None

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autots.duckdb")
TABLE_NAME = "nbjb"
DEFAULT_DELAY = 0.3  # 每股抓取间隔（秒），避免触发东财限流

API_BASE = "https://datacenter.eastmoney.com/securities/api/data/get"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

STY = ",".join([
    "SECURITY_CODE", "SECURITY_NAME_ABBR", "REPORTDATE", "REPORTDATEWZ", "REPORTDATEYW",
    "BASIC_EPS", "TOTAL_OPERATE_INCOME", "TOTAL_OPERATE_INCOME_TQ",
    "PARENT_NETPROFIT", "PARENT_NETPROFIT_TQ", "NOTICE_DATE",
])

# 入库字段（顺序与 fetch_stock 返回的元组一致）
COLS = [
    "code", "name", "report_date", "report_q", "report_label",
    "eps", "total_operate_income", "total_operate_income_yoy",
    "parent_netprofit", "parent_netprofit_yoy", "notice_date",
]


def _require_duckdb():
    if duckdb is None:
        raise ImportError("缺少 duckdb，请安装：python3 -m pip install duckdb")


def _to_date(s):
    """'2026-06-30 00:00:00' -> '2026-06-30'；空值/无日期返回 None（入库为 NULL）。"""
    d = (s or "").split(" ")[0]
    return d or None


def fetch_stock(code):
    """抓单只股票业绩报告，返回行列表。"""
    url = (
        f"{API_BASE}?type=RPT_LICO_FN_CPD_BB&source=DataCenter&client=WAP"
        f"&sty={STY}&p=1&ps=200&sr=-1&st=REPORTDATE"
        f"&filter=(SECURITY_CODE=%22{code}%22)"
    )
    try:
        r = subprocess.run(
            ["curl", "-s", "-f", "-x", "http://127.0.0.1:7897", url,
             "-H", f"User-Agent: {UA}", "-H", "Accept: application/json"],
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
        rd = _to_date(rec.get("REPORTDATE"))
        if not rd:
            continue  # 无报告期的记录跳过（主键必需）
        rows.append((
            code,
            rec.get("SECURITY_NAME_ABBR"),
            rd,
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


def _create_table(con):
    """建 nbjb 表（幂等）。"""
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


def ingest(rows, db_path=DEFAULT_DB_PATH):
    """UPSERT 到 DuckDB 的 nbjb 表。"""
    _require_duckdb()
    con = duckdb.connect(db_path)
    try:
        _create_table(con)
        cols = ", ".join(COLS)
        placeholders = ", ".join(["?"] * len(COLS))
        con.executemany(
            f"INSERT OR REPLACE INTO {TABLE_NAME} ({cols}) VALUES ({placeholders})",
            rows,
        )
    finally:
        con.close()
    return len(rows)


def load_stock_list(db_path=DEFAULT_DB_PATH, exclude_st=True):
    """从 stock_list 表读取 (symbol, name)。exclude_st=True 时排除名称含 ST 的股票。"""
    _require_duckdb()
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            "SELECT symbol, COALESCE(name, '') FROM stock_list ORDER BY symbol"
        ).fetchall()
    finally:
        con.close()
    entries = [(str(s), n or "") for s, n in rows]
    if exclude_st:
        entries = [e for e in entries if "ST" not in e[1].upper()]
    return entries


def write_failed(failed):
    """失败清单写文件，返回路径；无失败返回 None。"""
    if not failed:
        return None
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(temp_dir, f"nbjb_failed_{stamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("code\terror\n")
        for code in sorted(failed):
            f.write(f"{code}\t{failed[code]}\n")
    return path


def merge_stages(pattern, db_path=DEFAULT_DB_PATH):
    """把暂存 DuckDB 文件统一合并入主库 nbjb 表；已合并文件加 .merged 后缀。

    返回 (合并总行数, 合并文件数)。
    """
    import glob
    files = sorted(f for f in glob.glob(pattern) if not f.endswith(".merged"))
    if not files:
        raise ValueError(f"未找到暂存文件: {pattern}")
    _require_duckdb()
    con = duckdb.connect(db_path)
    total = 0
    try:
        _create_table(con)
        cols = ", ".join(COLS)
        for f in files:
            con.execute(f"ATTACH '{f}' AS stage (READ_ONLY)")
            stage_tables = {r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_catalog = 'stage'"
            ).fetchall()}
            if TABLE_NAME not in stage_tables:
                con.execute("DETACH stage")
                os.rename(f, f + ".merged")
                print(f"[业绩] 合并 {f}: 0 行（空暂存）")
                continue
            n = con.execute(f"SELECT COUNT(*) FROM stage.{TABLE_NAME}").fetchone()[0]
            con.execute(
                f"INSERT OR REPLACE INTO {TABLE_NAME} ({cols}) "
                f"SELECT {cols} FROM stage.{TABLE_NAME}"
            )
            con.execute("DETACH stage")
            os.rename(f, f + ".merged")
            total += n
            print(f"[业绩] 合并 {f}: {n} 行")
    finally:
        con.close()
    print(f"[业绩] 合并完成：{len(files)} 个文件，共 {total} 行 -> {db_path}")
    return total, len(files)


def run(codes=None, db_path=None, stage_path=None, exclude_st=True, delay=DEFAULT_DELAY):
    """供外部调用的入口。

    参数：
        codes: 股票代码列表；为 None 时从 stock_list 表读取在市股票（stock_list 模式）
        db_path: 主库 DuckDB 文件路径
        stage_path: 暂存 DuckDB 路径；"auto" 自动生成 download/temp/nbjb_stage_<时间戳>_<pid>.duckdb；
                    设置后数据写入暂存库而非主库，事后用 merge_stages 统一合并
        exclude_st: stock_list 模式下排除名称含 ST 的股票（默认 True）
        delay: 每股抓取间隔秒数（限速防封）
    返回：
        dict 包含 rows_count, db_path, series, failed_path, stage_path 等
    """
    db_path = db_path or DEFAULT_DB_PATH

    if stage_path == "auto":
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stage_path = os.path.join(temp_dir, f"nbjb_stage_{ts}_{os.getpid()}.duckdb")
    if stage_path:
        print(f"[业绩] 暂存模式：写入 {stage_path}，事后用 --merge 合并入主库")
    ingest_db = stage_path or db_path

    if isinstance(codes, str):
        codes = [c.strip() for c in codes.split(",") if c.strip()]

    if codes is None:
        entries_all = load_stock_list(db_path, exclude_st=False)
        n_st = sum(1 for e in entries_all if "ST" in e[1].upper())
        codes = [s for s, n in entries_all if not exclude_st or "ST" not in n.upper()]
        print(f"[业绩] 从 stock_list 读取 {len(entries_all)} 只，排除 ST {n_st} 只，待抓 {len(codes)} 只")
    elif not codes:
        raise ValueError("未提供股票代码")

    series = {}
    failed = {}
    success = set()
    total_rows = 0

    for idx, code in enumerate(codes, 1):
        print(f"[业绩] {idx}/{len(codes)} 抓取 {code} ...")
        try:
            rows = fetch_stock(code)
        except Exception as e:
            failed[code] = str(e)
            print(f"[业绩] {code} 失败: {e}", file=sys.stderr)
            time.sleep(delay)
            continue
        if not rows:
            failed[code] = "无数据"
            print(f"[业绩] {code} 无数据", file=sys.stderr)
            time.sleep(delay)
            continue
        n = ingest(rows, db_path=ingest_db)
        total_rows += n
        success.add(code)
        d0 = rows[0][2]
        d1 = rows[-1][2]
        series[code] = {"n": n, "first": d0, "last": d1}
        if idx % 100 == 0 or idx == len(codes):
            print(f"[业绩] 进度 {idx}/{len(codes)}  成功 {len(success)}  失败 {len(failed)}  累计入库 {total_rows} 行")
        time.sleep(delay)

    failed_path = write_failed(failed)

    print("\n[业绩] 入库概览")
    for code in sorted(series)[:20]:
        s = series[code]
        print(f"  {code}: n={s['n']:3d}  {s['first']} ~ {s['last']}")
    if len(series) > 20:
        print(f"  ... 共 {len(series)} 只成功")
    print(f"[业绩] 成功 {len(success)} 只，失败 {len(failed)} 只")
    if failed_path:
        print(f"[业绩] 失败清单: {failed_path}")

    result = {
        "rows_count": total_rows,
        "db_path": db_path,
        "series": series,
        "failed_path": failed_path,
        "stage_path": stage_path,
    }
    if failed:
        result["failed"] = failed
        if not series:
            raise RuntimeError(f"全部失败: {failed}")
    return result


def main():
    parser = argparse.ArgumentParser(description="东方财富业绩报告抓取入库（EPS/营收/净利润）")
    parser.add_argument("--codes", default=None, help="逗号分隔股票代码，如 300999,601318；不传则从 stock_list 表读取全部")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB 文件路径")
    parser.add_argument("--stage", default=None,
                        help="暂存 DuckDB 路径；auto 自动生成 download/temp/nbjb_stage_<时间戳>_<pid>.duckdb；不设则直入主库")
    parser.add_argument("--merge", default=None,
                        help="合并模式：暂存文件 glob，如 \"download/temp/nbjb_stage_*.duckdb\"，合并入 --db 后退出")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"每股间隔秒数，默认 {DEFAULT_DELAY}")
    parser.add_argument("--include-st", action="store_true", help="stock_list 模式下不排除 ST 股票")
    args = parser.parse_args()

    if args.merge:
        merge_stages(args.merge, db_path=args.db)
        return

    run(
        codes=args.codes,
        db_path=args.db,
        stage_path=args.stage,
        exclude_st=not args.include_st,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
