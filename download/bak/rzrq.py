# -*- coding: utf-8 -*-
"""东方财富个股融资融券抓取入库模块（日频）。

可独立运行，也可由 download_all.py 调用 run()。

用法（独立运行）：
    python -m download.rzrq                          # 从 stock_list 抓取全部 A 股两融明细
    python -m download.rzrq --codes 688223,300999    # 指定股票代码
    python -m download.rzrq --stage auto             # 暂存模式，事后 --merge 合并入主库

模式说明：
    * 不传 --codes 时，从 DuckDB stock_list 表读取全部在市股票（自动排除名称含 ST 的）。
    * 每股按主库 rzrq 已有最大日期做增量：翻页到 <= 该日期的旧数据即停，只补新交易日。
      因此首次全量（库空，翻到底）与每日增量走同一条路径，UPSERT 可重跑、可断点续抓。
    * 失败清单写入 download/temp/rzrq_failed_<时间戳>.txt，重跑即可补抓。

暂存模式（--stage）：
    数据不入主库，改写入暂存 DuckDB（表结构与主库 rzrq 完全相同），
    供多个并行任务各自生成暂存文件后，用 --merge 统一合并入主库：
        python -m download.rzrq --stage auto --codes 000001,600000
        python -m download.rzrq --merge "download/temp/rzrq_stage_*.duckdb"
    合并成功的暂存文件自动重命名加 .merged 后缀，防止重复合并。
    注意：暂存模式下增量起点仍读主库 rzrq 表，--merge 之后再跑增量才能续上。

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
TABLE_NAME = "rzrq"
PAGE_SIZE = 500
DEFAULT_DELAY = 0.3  # 每股抓取间隔（秒），避免触发东财限流

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

# 入库字段（顺序与 fetch_stock 返回的元组一致）
COLS = ["code", "name", "date"] + FIELD_KEYS


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
            ["curl", "-s", "-f", "-x", "http://127.0.0.1:7897", url,
             "-H", f"User-Agent: {UA}", "-H", "Accept: application/json"],
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


def fetch_stock(code, min_date=None):
    """抓单只股票两融数据（自动翻页）。

    min_date 给定时（如 "2026-09-22"），按 TRADE_DATE 倒序翻到 <= min_date 的旧数据即停，
    只返回更新日期，用于每日增量。
    """
    rows = []
    page = 1
    pages = 1
    while page <= pages:
        data, pages = _fetch_page(code, page)
        if not data:
            break
        for rec in data:
            d = _to_date(rec.get("TRADE_DATE"))
            if min_date and d and d <= min_date:
                rows.sort(key=lambda r: (r[0], r[2]))
                return rows
            rows.append(tuple(
                [code, rec.get("SECURITY_NAME_ABBR"), d]
                + [rec.get(k) for k in FIELD_KEYS]
            ))
        page += 1
    rows.sort(key=lambda r: (r[0], r[2]))
    return rows


def _create_table(con):
    """建 rzrq 表（幂等）。"""
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


def ingest(rows, db_path=DEFAULT_DB_PATH):
    """UPSERT 到 DuckDB 的 rzrq 表。"""
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


def get_max_dates(db_path=DEFAULT_DB_PATH):
    """返回主库 rzrq 已有数据的 {code: 最大日期字符串}，用于增量翻页起点。"""
    _require_duckdb()
    con = duckdb.connect(db_path, read_only=True)
    try:
        try:
            rows = con.execute(
                f"SELECT code, MAX(date) FROM {TABLE_NAME} GROUP BY code"
            ).fetchall()
        except Exception:
            return {}  # 表不存在
        return {
            str(c): (d.isoformat() if isinstance(d, datetime.date) else str(d))
            for c, d in rows
        }
    finally:
        con.close()


def write_failed(failed):
    """失败清单写文件，返回路径；无失败返回 None。"""
    if not failed:
        return None
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(temp_dir, f"rzrq_failed_{stamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("code\terror\n")
        for code in sorted(failed):
            f.write(f"{code}\t{failed[code]}\n")
    return path


def merge_stages(pattern, db_path=DEFAULT_DB_PATH):
    """把暂存 DuckDB 文件统一合并入主库 rzrq 表；已合并文件加 .merged 后缀。

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
                print(f"[两融] 合并 {f}: 0 行（空暂存）")
                continue
            n = con.execute(f"SELECT COUNT(*) FROM stage.{TABLE_NAME}").fetchone()[0]
            con.execute(
                f"INSERT OR REPLACE INTO {TABLE_NAME} ({cols}) "
                f"SELECT {cols} FROM stage.{TABLE_NAME}"
            )
            con.execute("DETACH stage")
            os.rename(f, f + ".merged")
            total += n
            print(f"[两融] 合并 {f}: {n} 行")
    finally:
        con.close()
    print(f"[两融] 合并完成：{len(files)} 个文件，共 {total} 行 -> {db_path}")
    return total, len(files)


def run(codes=None, db_path=None, stage_path=None, exclude_st=True, delay=DEFAULT_DELAY):
    """供外部调用的入口。

    参数：
        codes: 股票代码列表；为 None 时从 stock_list 表读取在市股票（stock_list 模式）
        db_path: 主库 DuckDB 文件路径
        stage_path: 暂存 DuckDB 路径；"auto" 自动生成 download/temp/rzrq_stage_<时间戳>_<pid>.duckdb；
                    设置后数据写入暂存库而非主库，事后用 merge_stages 统一合并；增量起点仍读主库
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
        stage_path = os.path.join(temp_dir, f"rzrq_stage_{ts}_{os.getpid()}.duckdb")
    if stage_path:
        print(f"[两融] 暂存模式：写入 {stage_path}，事后用 --merge 合并入主库")
    ingest_db = stage_path or db_path

    if isinstance(codes, str):
        codes = [c.strip() for c in codes.split(",") if c.strip()]

    stock_list_mode = codes is None
    if stock_list_mode:
        entries_all = load_stock_list(db_path, exclude_st=False)
        n_st = sum(1 for e in entries_all if "ST" in e[1].upper())
        codes = [s for s, n in entries_all if not exclude_st or "ST" not in n.upper()]
        print(f"[两融] 从 stock_list 读取 {len(entries_all)} 只，排除 ST {n_st} 只，待抓 {len(codes)} 只")
        max_dates = get_max_dates(db_path)  # 增量起点始终读主库
    else:
        if not codes:
            raise ValueError("未提供股票代码")
        max_dates = get_max_dates(db_path)  # --codes 同样按主库最大日期增量

    series = {}
    failed = {}
    success = set()
    no_new = 0
    total_rows = 0

    for idx, code in enumerate(codes, 1):
        min_date = max_dates.get(code)
        print(f"[两融] {idx}/{len(codes)} 抓取 {code} ...")
        try:
            rows = fetch_stock(code, min_date=min_date)
        except Exception as e:
            failed[code] = str(e)
            print(f"[两融] {code} 失败: {e}", file=sys.stderr)
            time.sleep(delay)
            continue
        if not rows:
            # 库中已有该股数据且无新交易日 = 已最新，不算失败
            if code in max_dates:
                no_new += 1
                continue
            failed[code] = "无数据"
            print(f"[两融] {code} 无数据", file=sys.stderr)
            time.sleep(delay)
            continue
        n = ingest(rows, db_path=ingest_db)
        total_rows += n
        success.add(code)
        series[code] = {"n": n, "first": rows[0][2], "last": rows[-1][2]}
        if idx % 100 == 0 or idx == len(codes):
            print(f"[两融] 进度 {idx}/{len(codes)}  成功 {len(success)}  失败 {len(failed)}  无新数据 {no_new}  累计入库 {total_rows} 行")
        time.sleep(delay)

    failed_path = write_failed(failed)

    print("\n[两融] 入库概览")
    for code in sorted(series)[:20]:
        s = series[code]
        print(f"  {code}: n={s['n']:5d}  {s['first']} ~ {s['last']}")
    if len(series) > 20:
        print(f"  ... 共 {len(series)} 只成功")
    print(f"[两融] 成功 {len(success)} 只，失败 {len(failed)} 只，无新数据 {no_new} 只")
    if failed_path:
        print(f"[两融] 失败清单: {failed_path}")

    result = {
        "rows_count": total_rows,
        "db_path": db_path,
        "series": series,
        "failed_path": failed_path,
        "stage_path": stage_path,
    }
    if failed:
        result["failed"] = failed
        if not series and not no_new:
            raise RuntimeError(f"全部失败: {failed}")
    return result


def main():
    parser = argparse.ArgumentParser(description="东方财富个股融资融券抓取入库（日频）")
    parser.add_argument("--codes", default=None, help="逗号分隔股票代码，如 688223,300999；不传则从 stock_list 表读取全部")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB 文件路径")
    parser.add_argument("--stage", default=None,
                        help="暂存 DuckDB 路径；auto 自动生成 download/temp/rzrq_stage_<时间戳>_<pid>.duckdb；不设则直入主库")
    parser.add_argument("--merge", default=None,
                        help="合并模式：暂存文件 glob，如 \"download/temp/rzrq_stage_*.duckdb\"，合并入 --db 后退出")
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
