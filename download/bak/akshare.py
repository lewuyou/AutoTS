# -*- coding: utf-8 -*-
"""AKShare 腾讯证券 A 股历史行情抓取入库模块。

通过 akshare 的 stock_zh_a_hist_tx 接口拉取日频 K 线数据，
支持不复权/前复权/后复权三种方式。

可独立运行，也可由 download_all.py 调用 run()。

用法（独立运行）：
    python -m download.akshare                          # 从 stock_list 全量/增量抓取全部 A 股
    python -m download.akshare --codes 000001,600000    # 指定股票
    python -m download.akshare --start 2020-01-01       # 指定起始日期
    python -m download.akshare --adjust qfq             # 前复权
    python -m download.akshare --adjust hfq             # 后复权

模式说明：
    * 不传 --codes/--codes-file 时，从 DuckDB stock_list 表读取全部在市股票（自动排除名称含 ST 的），
      每股起始日期 = max(list_date, 库中该股同复权方式最大日期 + 1 天)。
      因此日常增量与首次全量走同一条路径，UPSERT 可重跑、可断点续抓。
    * 失败清单写入 download/temp/akshare_failed_<时间戳>.txt，重跑即可补抓。

暂存模式（--stage）：
    数据不入主库，改写入暂存 DuckDB（表结构与主库 akshare_tx 完全相同），
    供多个并行任务各自生成暂存文件后，用 --merge 统一合并入主库：
        python -m download.akshare --stage auto --codes 000001,600000
        python -m download.akshare --merge "download/temp/akshare_stage_*.duckdb"
    合并成功的暂存文件自动重命名加 .merged 后缀，防止重复合并。
    注意：暂存模式下增量起点仍读主库 akshare_tx 表，--merge 之后再跑增量才能续上。

依赖：
    pip install akshare duckdb

入库表 akshare_tx 字段含义：
    symbol      股票代码（如 "000001"，自动补全市场前缀 sz/sh）
    date        交易日
    open        开盘价（元）
    close       收盘价（元）
    high        最高价（元）
    low         最低价（元）
    volume      成交量（股；腾讯接口原始单位为手，入库时已 ×100 换算）
    turnover    换手率（小数）
    amount      成交额（元）
    adjust      复权方式（""=不复权, qfq=前复权, hfq=后复权）
"""

import argparse
import datetime
import os
import sys
import time

try:
    import akshare as ak
except ImportError:
    ak = None

try:
    import duckdb
except ImportError:
    duckdb = None

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autots.duckdb")
TABLE_NAME = "akshare_tx"
SHARES_PER_LOT = 100  # 腾讯接口 volume 单位为手，1 手 = 100 股
DEFAULT_DELAY = 1.0   # 每股抓取间隔（秒），避免触发腾讯限流
RETRY_TIMES = 2
RETRY_BACKOFF = 3.0


def _require_akshare():
    if ak is None:
        raise ImportError("缺少 akshare，请安装：python3 -m pip install akshare")


def _require_duckdb():
    if duckdb is None:
        raise ImportError("缺少 duckdb，请安装：python3 -m pip install duckdb")


def code_to_symbol(code):
    """A 股代码 -> 腾讯证券 symbol。6/9 开头上交所 sh，0/2/3 开头深交所 sz，4/8 开头北交所 bj。"""
    code = str(code).strip()
    if code.startswith(("sh", "sz", "bj")):
        return code
    if code[0] in ("6", "9"):
        return f"sh{code}"
    if code[0] in ("0", "2", "3"):
        return f"sz{code}"
    if code[0] in ("4", "8"):
        return f"bj{code}"
    return f"sz{code}"


class EmptyDataError(RuntimeError):
    """接口返回空数据（区间内无交易日或品种无数据）。"""


def fetch_hist(symbol, start_date, end_date, adjust="", timeout=30, retries=RETRY_TIMES):
    """调用 akshare 拉取历史行情，返回 DataFrame；空数据不重试，网络等异常按次数重试。"""
    _require_akshare()
    last_err = None
    for attempt in range(retries + 1):
        try:
            df = ak.stock_zh_a_hist_tx(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
                timeout=timeout,
            )
            if df is None or df.empty:
                raise EmptyDataError(f"{symbol}: 未返回数据")
            return df
        except EmptyDataError:
            raise
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
    raise last_err


def df_to_rows(code, df, adjust=""):
    """DataFrame 转成入库行 (symbol, date, open, close, high, low, volume, turnover, amount, adjust)。

    volume 原始单位为手，此处 ×100 统一为股。
    """
    rows = []
    for _, r in df.iterrows():
        rows.append((
            code,
            r["date"],
            float(r["open"]),
            float(r["close"]),
            float(r["high"]),
            float(r["low"]),
            float(r["volume"]) * SHARES_PER_LOT,
            float(r["turnover"]),
            float(r["amount"]),
            adjust,
        ))
    return rows


def ingest(rows, db_path=DEFAULT_DB_PATH):
    """UPSERT 到 DuckDB 的 akshare_tx 表。"""
    _require_duckdb()
    con = duckdb.connect(db_path)
    try:
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                symbol TEXT,
                date DATE,
                open DOUBLE,
                close DOUBLE,
                high DOUBLE,
                low DOUBLE,
                volume DOUBLE,
                turnover DOUBLE,
                amount DOUBLE,
                adjust TEXT,
                PRIMARY KEY (symbol, date, adjust)
            )
            """
        )
        con.executemany(
            f"INSERT OR REPLACE INTO {TABLE_NAME} (symbol, date, open, close, high, low, volume, turnover, amount, adjust) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    finally:
        con.close()
    return len(rows)


def write_csv(rows, base_dir, code, adjust=""):
    """按股票代码写 CSV。"""
    suffix = f"_{adjust}" if adjust else ""
    path = os.path.join(base_dir, f"akshare_tx_{code}{suffix}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        f.write("date,open,close,high,low,volume,turnover,amount\n")
        for _, d, o, c, h, l, v, t, a, _ in rows:
            f.write(f"{d},{o},{c},{h},{l},{v},{t},{a}\n")
    return path


def load_stock_list(db_path=DEFAULT_DB_PATH, exclude_st=True):
    """从 stock_list 表读取 (symbol, list_date, name)，list_date 缺失的用 1990-01-01。

    exclude_st=True 时排除名称含 ST 的股票（含 *ST）。
    """
    _require_duckdb()
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            "SELECT symbol, COALESCE(list_date, DATE '1990-01-01'), COALESCE(name, '') FROM stock_list ORDER BY symbol"
        ).fetchall()
    finally:
        con.close()
    entries = [(str(s), d, n or "") for s, d, n in rows]
    if exclude_st:
        entries = [e for e in entries if "ST" not in e[2].upper()]
    return entries


def get_max_dates(db_path, adjust, db_con=None):
    """返回 {symbol: 库中该复权方式的最大日期}，用于增量起始日计算。"""
    _require_duckdb()
    own = db_con is None
    con = db_con or duckdb.connect(db_path, read_only=True)
    try:
        try:
            rows = con.execute(
                f"SELECT symbol, MAX(date) FROM {TABLE_NAME} WHERE adjust = ? GROUP BY symbol",
                [adjust],
            ).fetchall()
        except Exception:
            return {}  # 表不存在
        return {str(s): d for s, d in rows}
    finally:
        if own:
            con.close()


def write_failed(failed, csv_dir):
    """失败清单写文件，返回路径；无失败返回 None。"""
    if not failed:
        return None
    os.makedirs(csv_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(csv_dir, f"akshare_failed_{stamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("symbol\terror\n")
        for code in sorted(failed):
            f.write(f"{code}\t{failed[code]}\n")
    return path


def merge_stages(pattern, db_path=DEFAULT_DB_PATH):
    """把暂存 DuckDB 文件统一合并入主库 akshare_tx 表；已合并文件加 .merged 后缀。

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
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                symbol TEXT,
                date DATE,
                open DOUBLE,
                close DOUBLE,
                high DOUBLE,
                low DOUBLE,
                volume DOUBLE,
                turnover DOUBLE,
                amount DOUBLE,
                adjust TEXT,
                PRIMARY KEY (symbol, date, adjust)
            )
            """
        )
        for f in files:
            con.execute(f"ATTACH '{f}' AS stage (READ_ONLY)")
            stage_tables = {r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_catalog = 'stage'"
            ).fetchall()}
            if TABLE_NAME not in stage_tables:
                con.execute("DETACH stage")
                os.rename(f, f + ".merged")
                print(f"[AKShare] 合并 {f}: 0 行（空暂存）")
                continue
            n = con.execute(f"SELECT COUNT(*) FROM stage.{TABLE_NAME}").fetchone()[0]
            con.execute(
                f"INSERT OR REPLACE INTO {TABLE_NAME} (symbol, date, open, close, high, low, volume, turnover, amount, adjust) "
                f"SELECT symbol, date, open, close, high, low, volume, turnover, amount, adjust FROM stage.{TABLE_NAME}"
            )
            con.execute("DETACH stage")
            os.rename(f, f + ".merged")
            total += n
            print(f"[AKShare] 合并 {f}: {n} 行")
    finally:
        con.close()
    print(f"[AKShare] 合并完成：{len(files)} 个文件，共 {total} 行 -> {db_path}")
    return total, len(files)


def run(codes=None, start=None, end=None, adjust="hfq", db_path=None, no_csv=False, csv_dir=None,
        codes_file=None, delay=DEFAULT_DELAY, exclude_st=True, stage_path=None):
    """供外部调用的入口。

    参数：
        codes: 股票代码列表；为 None 且未指定 codes_file 时，从 stock_list 表读取在市股票
        start: 起始日期 YYYY-MM-DD；指定股票时默认当年 1 月 1 日；
               stock_list 模式下为每股 max(list_date, 库中最大日期+1)，此参数仅作下限
        end: 结束日期 YYYY-MM-DD，默认今天
        adjust: 复权方式，""=不复权, qfq=前复权, hfq=后复权
        db_path: DuckDB 文件路径，默认模块目录下 autots.duckdb
        no_csv: 是否跳过 CSV 导出
        csv_dir: CSV/失败清单输出目录，默认模块目录下 temp/
        codes_file: CSV 文件路径，包含 symbol 列的股票代码列表
        delay: 每只股票之间的间隔秒数（限速防封）
        exclude_st: stock_list 模式下排除名称含 ST 的股票（默认 True）
        stage_path: 暂存 DuckDB 路径；"auto" 自动生成 download/temp/akshare_stage_<时间戳>_<pid>.duckdb；
                    设置后数据写入暂存库而非主库，事后用 merge_stages 统一合并；
                    增量起点仍读主库 akshare_tx
    返回：
        dict 包含 rows_count, db_path, csv_paths, series, failed_path, stage_path 等
    """
    db_path = db_path or DEFAULT_DB_PATH
    csv_dir = csv_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")

    if stage_path == "auto":
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stage_path = os.path.join(csv_dir, f"akshare_stage_{ts}_{os.getpid()}.duckdb")
    if stage_path:
        print(f"[AKShare] 暂存模式：写入 {stage_path}，事后用 --merge 合并入主库")
    ingest_db = stage_path or db_path

    # 从 CSV 文件读取股票代码
    if codes is None and codes_file:
        import pandas as pd
        df_codes = pd.read_csv(codes_file)
        if "symbol" not in df_codes.columns:
            raise ValueError(f"CSV 文件 {codes_file} 缺少 symbol 列")
        codes = df_codes["symbol"].astype(str).tolist()
        print(f"[AKShare] 从 {codes_file} 读取 {len(codes)} 只股票")
        if not codes:
            raise ValueError("CSV 文件中无股票代码")

    if isinstance(codes, str):
        codes = [c.strip() for c in codes.split(",") if c.strip()]

    today = datetime.date.today()
    end = (end or today.strftime("%Y%m%d")).replace("-", "")
    end_date = datetime.datetime.strptime(end, "%Y%m%d").date()

    # stock_list 模式：每股独立起始日 = max(list_date, 主库最大日期+1)
    stock_list_mode = codes is None
    if stock_list_mode:
        entries_all = load_stock_list(db_path, exclude_st=False)
        n_st = sum(1 for e in entries_all if "ST" in e[2].upper())
        entries = [(s, d) for s, d, n in entries_all if not exclude_st or "ST" not in n.upper()]
        print(f"[AKShare] 从 stock_list 读取 {len(entries_all)} 只，排除 ST {n_st} 只，待抓 {len(entries)} 只")
        max_dates = get_max_dates(db_path, adjust)  # 增量起点始终读主库
        lower = start.replace("-", "") if start else None
    else:
        if not codes:
            raise ValueError("未提供股票代码")
        start = (start or f"{today.year}0101").replace("-", "")
        entries = [(c, None) for c in codes]
        max_dates = {}
        lower = None

    if not no_csv:
        os.makedirs(csv_dir, exist_ok=True)

    total_rows = 0
    csv_paths = {}
    series = {}
    failed = {}
    skipped = 0
    no_new = 0

    for idx, (code, list_date) in enumerate(entries, 1):
        symbol = code_to_symbol(code)
        adjust_label = adjust or "不复权"

        if stock_list_mode:
            sdate = list_date
            last = max_dates.get(code)
            if last is not None and last + datetime.timedelta(days=1) > sdate:
                sdate = last + datetime.timedelta(days=1)
            if lower:
                ldate = datetime.datetime.strptime(lower, "%Y%m%d").date()
                if ldate > sdate:
                    sdate = ldate
            if sdate > end_date:
                skipped += 1
                continue
            start_i = sdate.strftime("%Y%m%d")
        else:
            start_i = start

        try:
            df = fetch_hist(symbol, start_i, end, adjust=adjust)
        except EmptyDataError as e:
            # 库中已有该股数据时空返回 = 没有新交易日，不算失败
            if stock_list_mode and code in max_dates:
                no_new += 1
                continue
            failed[code] = str(e)
            print(f"[AKShare] {code} 无数据: {e}", file=sys.stderr)
            time.sleep(delay)
            continue
        except Exception as e:
            failed[code] = str(e)
            print(f"[AKShare] {code} 失败: {e}", file=sys.stderr)
            time.sleep(delay)
            continue
        rows = df_to_rows(code, df, adjust=adjust)
        if not rows:
            failed[code] = "无数据"
            print(f"[AKShare] {code} 无数据", file=sys.stderr)
            time.sleep(delay)
            continue
        n = ingest(rows, db_path=ingest_db)
        total_rows += n
        if not no_csv:
            csv_paths[code] = write_csv(rows, csv_dir, code, adjust)
        closes = [r[3] for r in rows]
        series[code] = {
            "n": len(rows),
            "first": rows[0][1],
            "last": rows[-1][1],
            "mean_close": sum(closes) / len(closes) if closes else 0,
        }
        if idx % 100 == 0 or idx == len(entries):
            print(f"[AKShare] 进度 {idx}/{len(entries)}  成功 {len(series)}  失败 {len(failed)}  无新数据 {no_new}  累计入库 {total_rows} 行")
        else:
            print(f"[AKShare] {code} ({symbol}) {adjust_label} 入库 {n} 行  {rows[0][1]} ~ {rows[-1][1]}")
        time.sleep(delay)

    failed_path = write_failed(failed, csv_dir)

    print("\n[AKShare] 入库概览")
    for code in sorted(series)[:20]:
        s = series[code]
        print(f"  {code}: n={s['n']:5d}  mean_close={s['mean_close']:10.2f}  {s['first']} ~ {s['last']}")
    if len(series) > 20:
        print(f"  ... 共 {len(series)} 只成功")
    print(f"[AKShare] 成功 {len(series)} 只，失败 {len(failed)} 只，无新数据 {no_new} 只，跳过(已最新) {skipped} 只")
    if failed_path:
        print(f"[AKShare] 失败清单: {failed_path}")

    result = {
        "rows_count": total_rows,
        "db_path": db_path,
        "csv_paths": csv_paths or None,
        "series": series,
        "failed_path": failed_path,
        "stage_path": stage_path,
    }
    if failed:
        result["failed"] = failed
        if not series and not no_new and not skipped:
            raise RuntimeError(f"全部失败: {failed}")
    return result


def main():
    parser = argparse.ArgumentParser(description="AKShare 腾讯证券 A 股历史行情抓取入库")
    parser.add_argument("--codes", default=None, help="逗号分隔股票代码，如 000001,600000；不传则从 stock_list 表读取全部")
    parser.add_argument("--codes-file", default=None, help="CSV 文件路径，包含 symbol 列的股票代码列表")
    parser.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD（指定股票时默认当年 1 月 1 日；stock_list 模式作下限）")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--adjust", default="hfq", choices=["", "qfq", "hfq"],
                        help="复权方式：\"\"=不复权, qfq=前复权, hfq=后复权（默认）")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB 文件路径")
    parser.add_argument("--no-csv", action="store_true", help="不写 CSV，只入库")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"每股间隔秒数，默认 {DEFAULT_DELAY}")
    parser.add_argument("--include-st", action="store_true", help="stock_list 模式下不排除 ST 股票")
    parser.add_argument("--stage", default=None,
                        help="暂存 DuckDB 路径；auto 自动生成 download/temp/akshare_stage_<时间戳>_<pid>.duckdb；不设则直入主库")
    parser.add_argument("--merge", default=None,
                        help="合并模式：暂存文件 glob，如 \"download/temp/akshare_stage_*.duckdb\"，合并入 --db 后退出")
    args = parser.parse_args()

    if args.merge:
        merge_stages(args.merge, db_path=args.db)
        return

    run(
        codes=args.codes,
        start=args.start,
        end=args.end,
        adjust=args.adjust,
        db_path=args.db,
        no_csv=args.no_csv,
        codes_file=args.codes_file,
        delay=args.delay,
        exclude_st=not args.include_st,
        stage_path=args.stage,
    )


if __name__ == "__main__":
    main()
