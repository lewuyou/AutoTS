# -*- coding: utf-8 -*-
"""TradingView 估值指标（PE/PB/PS/PCF）抓取入库模块。

从 TradingView 图表布局 acmWP4Eo 抓 4 个基本面指标（市盈率/市净率/市销率/市现率），
入库到 guzhi 表。复用 tradingview_study.py 的 WebSocket 抓取逻辑；symbol 存 6 位 A 股代码，
从 stock_list 表读取全部在市股票（含 ST），按 exchange 映射为 TradingView symbol：
    SH -> SSE:600000    SZ -> SZSE:000001

依赖：
    pip install duckdb websocket-client requests

前置（一次性）：
    1. cookie：download/tv/tv_cookie.txt（登录 cookie，含 sessionid）
    2. 加密 text：download/tv/tv_study_texts.json 里必须有本布局 4 个指标的 text，
       缺失时先跑  python -m download.tv.fetch_tv_study_texts --layout acmWP4Eo --headless

用法（独立运行）：
    python -m download.tv.tradingview_guzhi                        # 增量抓取：只抓 stock_list 中尚未入库的（含 ST），直入主库
    python -m download.tv.tradingview_guzhi --full                 # 全量重抓所有股票（覆盖已有数据）
    python -m download.tv.tradingview_guzhi --stage auto --log auto# 暂存模式 + 写日志，事后 --merge 合并
    python -m download.tv.tradingview_guzhi --codes 600000,000001  # 指定代码（沪+深）
    python -m download.tv.tradingview_guzhi --merge "download/temp/guzhi_stage_*.duckdb"
    python -m download.tv.tradingview_guzhi --no-proxy
    python -m download.tv.tradingview_guzhi --no-refresh-texts  # 跳过抓取前重新提取 text（用缓存）

暂存模式（--stage）：
    数据不入主库，改写入暂存 DuckDB（表结构与主库 guzhi 完全相同），
    供长时间抓取期间不占用主库锁、不影响其他抓取任务，事后用 --merge 统一合并：
        python -m download.tv.tradingview_guzhi --stage auto --log auto
        python -m download.tv.tradingview_guzhi --merge "download/temp/guzhi_stage_*.duckdb"
    合并成功的暂存文件自动重命名加 .merged 后缀，防止重复合并。

入库表 guzhi 字段含义：
    symbol     股票代码（6 位，如 "000001"，与 stock_list.symbol / gzfx.code 一致，可直接 JOIN）
    indicator  指标类型：pe=市盈率 pb=市净率 ps=市销率 pcf=市现率
    date       交易日（UTC 时间戳转北京时间）
    value      指标值（TradingView 基本面指标日频值；上市前/无数据为空，已过滤）
"""

import argparse
import datetime
import glob
import json
import os
import sys
import time

from . import fetch_tv_study_texts
from . import tradingview_study as ts

TABLE_NAME = "guzhi"
DEFAULT_LAYOUT = "acmWP4Eo"
DEFAULT_DB_PATH = ts.DEFAULT_DB_PATH
DEFAULT_COOKIE_FILE = ts.DEFAULT_COOKIE_FILE
DEFAULT_TEXTS_FILE = ts.DEFAULT_TEXTS_FILE
MAX_RETRY = 3        # 每股抓取最大轮数（连接失败重试）
RETRY_DELAY = 3      # 重试间隔秒
REFRESH_EVERY = 150  # 每抓 N 只刷新一次 auth_token（token 约 4 小时过期）
DEFAULT_DELAY = 5    # 每股间隔秒数（限速）

_DOWNLOAD_DIR = os.path.dirname(ts.DEFAULT_DB_PATH)
_TEMP_DIR = os.path.join(_DOWNLOAD_DIR, "temp")

# metainfo 里 "Internal$STD;Fund_xxx@..." 的 Fund_xxx -> 指标名
INDICATOR_MAP = {
    "Fund_price_earnings": "pe",
    "Fund_price_book": "pb",
    "Fund_price_sales": "ps",
    "Fund_price_cash_flow": "pcf",
}


def _setup_log(log_path):
    """把 stdout/stderr 同步 tee 到日志文件（供长任务 tail 看进度）。"""
    if not log_path:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    f = open(log_path, "a", encoding="utf-8")

    class Tee:
        def __init__(self, stream, file):
            self.stream = stream
            self.file = file

        def write(self, s):
            self.stream.write(s)
            self.file.write(s)
            self.file.flush()

        def flush(self):
            self.stream.flush()
            self.file.flush()

        def isatty(self):
            return getattr(self.stream, "isatty", lambda: False)()

        def fileno(self):
            return self.stream.fileno()

    sys.stdout = Tee(sys.stdout, f)
    sys.stderr = Tee(sys.stderr, f)
    return f


def study_indicator(metainfo):
    """从 metainfo 提取指标名：'Internal$STD;Fund_price_earnings@...' -> 'pe'；非基本面返回 None。"""
    if not metainfo:
        return None
    for part in metainfo.split(";"):
        name = part.split("@")[0]
        if name in INDICATOR_MAP:
            return INDICATOR_MAP[name]
    return None


def code_to_symbol(code, exchange=""):
    """6 位代码 + 交易所 -> TradingView symbol。SZ -> SZSE:，其余（SH/未知）-> SSE:。"""
    code = str(code).strip()
    if ":" in code:
        return code
    if str(exchange).upper() == "SZ":
        return f"SZSE:{code}"
    return f"SSE:{code}"


def load_stock_list(db_path=DEFAULT_DB_PATH):
    """从 stock_list 表读取 (symbol, exchange)，按总市值 total_mv 从大到小排序（缺失排最后），全部在市股票（含 ST）。"""
    ts._require()
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            "SELECT symbol, COALESCE(exchange, '') FROM stock_list ORDER BY total_mv DESC NULLS LAST, symbol"
        ).fetchall()
    finally:
        con.close()
    return [(str(s), str(e or "")) for s, e in rows]


def load_ingested_symbols(db_path):
    """查某 DuckDB 库 guzhi 表已入库的 symbol 集合；表不存在或库不存在返回空集。"""
    ts._require()
    if not os.path.exists(db_path):
        return set()
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if TABLE_NAME not in tables:
            return set()
        return {r[0] for r in con.execute(f"SELECT DISTINCT symbol FROM {TABLE_NAME}").fetchall()}
    finally:
        con.close()


def _codes_with_exchange(codes, db_path):
    """把传入的代码列表补上 exchange（查 stock_list，查不到则留空由 code_to_symbol 兜底）。"""
    ex_map = {s: e for s, e in load_stock_list(db_path)}
    out = []
    for c in codes:
        c = str(c).strip()
        if not c:
            continue
        out.append((c, ex_map.get(c, "")))
    return out


def _create_table(con):
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            symbol TEXT,
            indicator TEXT,
            date DATE,
            value DOUBLE,
            PRIMARY KEY (symbol, indicator, date)
        )
        """
    )


def ingest(rows, db_path=DEFAULT_DB_PATH):
    """UPSERT 到 DuckDB 的 guzhi 表。"""
    ts._require()
    import duckdb
    con = duckdb.connect(db_path)
    try:
        _create_table(con)
        con.executemany(
            f"INSERT OR REPLACE INTO {TABLE_NAME} (symbol, indicator, date, value) VALUES (?, ?, ?, ?)",
            rows,
        )
    finally:
        con.close()
    return len(rows)


def studies_to_rows(symbol_code, study_data, id_to_indicator, start=None, end=None):
    """把 {study_id: [(ts, [v...])]} 转成 (symbol, indicator, date, value) 行，取每个指标 plot0，过滤空值。"""
    rows = []
    seen = set()
    for st_id, pts in study_data.items():
        ind = id_to_indicator.get(st_id)
        if not ind:
            continue
        for t, vals in pts:
            d = ts.ts_to_date(t)
            if start and d < start:
                continue
            if end and d > end:
                continue
            v = vals[0] if vals else None
            if v is None or v == ts.EMPTY or (isinstance(v, float) and v >= ts.EMPTY):
                continue
            if isinstance(v, bool):
                continue
            key = (ind, d)
            if key in seen:
                continue
            seen.add(key)
            rows.append((symbol_code, ind, d, float(v)))
    return rows


def write_failed(failed):
    """失败清单写文件，返回路径；无失败返回 None。"""
    if not failed:
        return None
    os.makedirs(_TEMP_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_TEMP_DIR, f"guzhi_failed_{stamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("symbol\terror\n")
        for code in sorted(failed):
            f.write(f"{code}\t{failed[code]}\n")
    return path


def merge_stages(pattern, db_path=DEFAULT_DB_PATH):
    """把暂存 DuckDB 文件统一合并入主库 guzhi 表；已合并文件加 .merged 后缀。

    返回 (合并总行数, 合并文件数)。
    """
    files = sorted(f for f in glob.glob(pattern) if not f.endswith(".merged"))
    if not files:
        raise ValueError(f"未找到暂存文件: {pattern}")
    ts._require()
    import duckdb
    con = duckdb.connect(db_path)
    total = 0
    try:
        _create_table(con)
        cols = "symbol, indicator, date, value"
        for f in files:
            con.execute(f"ATTACH '{f}' AS stage (READ_ONLY)")
            stage_tables = {r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_catalog = 'stage'"
            ).fetchall()}
            if TABLE_NAME not in stage_tables:
                con.execute("DETACH stage")
                os.rename(f, f + ".merged")
                print(f"[估值TV] 合并 {f}: 0 行（空暂存）")
                continue
            n = con.execute(f"SELECT COUNT(*) FROM stage.{TABLE_NAME}").fetchone()[0]
            con.execute(
                f"INSERT OR REPLACE INTO {TABLE_NAME} ({cols}) "
                f"SELECT {cols} FROM stage.{TABLE_NAME}"
            )
            con.execute("DETACH stage")
            os.rename(f, f + ".merged")
            total += n
            print(f"[估值TV] 合并 {f}: {n} 行")
    finally:
        con.close()
    print(f"[估值TV] 合并完成：{len(files)} 个文件，共 {total} 行 -> {db_path}")
    return total, len(files)


def run(codes=None, layout_id=DEFAULT_LAYOUT, start=None, end=None,
        cookie_file=None, texts_file=None, db_path=None, stage_path=None,
        use_proxy=True, delay=DEFAULT_DELAY, log_path=None, full=False,
        refresh_texts=True):
    """供外部调用的入口。

    参数：
        codes: 股票代码列表；为 None 时从 stock_list 表读取全部在市股票（含 ST）
        layout_id: 布局编号，默认 acmWP4Eo（含 PE/PB/PS/PCF 四个基本面指标）
        start/end: 日期过滤 YYYY-MM-DD，None 抓全历史
        cookie_file: cookie 文件，默认 download/tv/tv_cookie.txt
        texts_file: 加密 text 缓存，默认 download/tv/tv_study_texts.json
        db_path: 主库 DuckDB 文件路径，默认 download/autots.duckdb
        stage_path: 暂存 DuckDB 路径；"auto" 自动生成 download/temp/guzhi_stage_<时间戳>_<pid>.duckdb；
                    设置后数据写暂存库而非主库，事后用 merge_stages 统一合并
        use_proxy: 是否走本机 7897 代理
        delay: 每股间隔秒数（限速）
        log_path: 日志文件路径；"auto" 自动生成 download/temp/guzhi_<时间戳>.log；
                  设置后 stdout/stderr 同步写日志（供 tail 看进度）
        full: True 时全量重抓所有股票（覆盖已有数据）；默认 False 增量抓取，
              只抓目标库中尚未入库的股票（暂存模式还叠加暂存库断点续抓）
        refresh_texts: True 时抓取前先跑 fetch_tv_study_texts 重新提取布局加密 text；
                       False 直接用 texts_file 里的缓存
    返回：
        dict 含 rows_count, db_path, series, failed_path, stage_path 等
    """
    _setup_log(log_path)
    ts._require()
    db_path = db_path or DEFAULT_DB_PATH

    if log_path == "auto":
        os.makedirs(_TEMP_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(_TEMP_DIR, f"guzhi_{stamp}.log")
        _setup_log(log_path)

    if stage_path == "auto":
        os.makedirs(_TEMP_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stage_path = os.path.join(_TEMP_DIR, f"guzhi_stage_{stamp}_{os.getpid()}.duckdb")
    if stage_path:
        print(f"[估值TV] 暂存模式：写入 {stage_path}，事后用 --merge 合并入主库")
    ingest_db = stage_path or db_path

    cookie_file = cookie_file or DEFAULT_COOKIE_FILE
    texts_file = texts_file or DEFAULT_TEXTS_FILE
    with open(cookie_file, encoding="utf-8") as f:
        cookie = f.read().strip()
    if refresh_texts:
        print(f"[估值TV] 重新提取布局 {layout_id} 的加密 text ...")
        fetch_tv_study_texts.fetch_texts(
            layout_id, cookie_file=cookie_file, out_file=texts_file,
            headless=True, use_proxy=use_proxy)
    with open(texts_file, encoding="utf-8") as f:
        texts = json.load(f)

    end = end or datetime.date.today().isoformat()

    print(f"[估值TV] 布局 {layout_id}，提取 auth_token + 指标配置 ...")
    auth_token, content = ts.load_layout_config(layout_id, cookie, use_proxy=use_proxy)
    studies = ts.parse_studies(content, texts)
    id_to_indicator = {}
    for st in studies:
        ind = study_indicator(st.get("metainfo", ""))
        if ind:
            id_to_indicator[st["id"]] = ind
    got_inds = sorted(set(id_to_indicator.values()))
    missing_inds = sorted(set(INDICATOR_MAP.values()) - set(id_to_indicator.values()))
    if not id_to_indicator:
        raise RuntimeError(
            "布局里没有 PE/PB/PS/PCF 指标，或 tv_study_texts.json 缺它们的加密 text。"
            "请先跑: python -m download.tv.fetch_tv_study_texts --layout %s --headless" % layout_id
        )
    if missing_inds:
        print(f"[估值TV] 警告：缺指标 {missing_inds} 的加密 text，本次只抓 {got_inds}", file=sys.stderr)
    print(f"[估值TV] 识别到指标: {got_inds}")

    if isinstance(codes, str):
        codes = [c.strip() for c in codes.split(",") if c.strip()]
    if codes:
        entries = _codes_with_exchange(codes, db_path)
        print(f"[估值TV] 指定代码 {len(entries)} 只")
    else:
        entries = load_stock_list(db_path)
        print(f"[估值TV] 从 stock_list 读取 {len(entries)} 只（含 ST）")
    if not entries:
        raise ValueError("未提供股票代码且 stock_list 表为空")

    series = {}
    failed = {}
    total_rows = 0
    skipped = 0

    # 增量抓取：默认跳过已入库股票，只抓没入库的；--full 全量重抓
    done_syms = set()
    if not full:
        if stage_path:
            # 暂存模式：跳过暂存库已有（断点续抓）+ 主库已有（合并过的）
            done_syms = load_ingested_symbols(ingest_db)
            if os.path.exists(db_path):
                done_syms |= load_ingested_symbols(db_path)
            if done_syms:
                print(f"[估值TV] 已入库 {len(done_syms)} 只（暂存库+主库），跳过只抓剩余")
        else:
            done_syms = load_ingested_symbols(db_path)
            if done_syms:
                print(f"[估值TV] 主库已有 {len(done_syms)} 只，跳过只抓未入库")

    for idx, (code, exchange) in enumerate(entries, 1):
        if code in done_syms:
            skipped += 1
            continue
        if idx > 1 and idx % REFRESH_EVERY == 1:
            print(f"[估值TV] 刷新 auth_token ...")
            auth_token, _ = ts.load_layout_config(layout_id, cookie, use_proxy=use_proxy)
        symbol = code_to_symbol(code, exchange)
        print(f"[估值TV] {idx}/{len(entries)} 抓取 {code} ({symbol}) ...")

        study_data = {}
        last_err = None
        for attempt in range(1, MAX_RETRY + 1):
            try:
                data, _missing = ts.fetch_symbol_studies(
                    symbol, studies, auth_token, layout_id, cookie, use_proxy=use_proxy)
            except Exception as e:
                last_err = e
                print(f"[估值TV] {code} 第{attempt}轮连接失败: {e}", file=sys.stderr)
                time.sleep(RETRY_DELAY)
                continue
            for sid, pts in data.items():
                study_data.setdefault(sid, []).extend(pts)
            break

        if not study_data:
            failed[code] = str(last_err) if last_err else "无数据"
            print(f"[估值TV] {code} 失败: {failed[code]}", file=sys.stderr)
        else:
            rows = studies_to_rows(code, study_data, id_to_indicator, start=start, end=end)
            if not rows:
                failed[code] = "无数据"
                print(f"[估值TV] {code} 无数据", file=sys.stderr)
            else:
                n = ingest(rows, db_path=ingest_db)
                total_rows += n
                inds = sorted({r[1] for r in rows})
                dates = [r[2] for r in rows]
                series[code] = {"n": len(rows), "inds": inds, "first": min(dates), "last": max(dates)}
                print(f"[估值TV] {code} 入库 {n} 行  指标{inds}  {min(dates)} ~ {max(dates)}")

        if idx % 50 == 0 or idx == len(entries):
            print(f"[估值TV] 进度 {idx}/{len(entries)}  成功 {len(series)}  失败 {len(failed)}  累计入库 {total_rows} 行")
        time.sleep(delay)

    failed_path = write_failed(failed)

    print("\n[估值TV] 入库概览")
    for code in sorted(series)[:20]:
        s = series[code]
        print(f"  {code}: n={s['n']:5d}  指标{s['inds']}  {s['first']} ~ {s['last']}")
    if len(series) > 20:
        print(f"  ... 共 {len(series)} 只")
    print(f"[估值TV] 成功 {len(series)} 只，失败 {len(failed)} 只，跳过 {skipped} 只")
    if failed_path:
        print(f"[估值TV] 失败清单: {failed_path}")

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
    parser = argparse.ArgumentParser(description="TradingView 估值指标抓取入库（PE/PB/PS/PCF）")
    parser.add_argument("--codes", default=None, help="逗号分隔股票代码；不传则从 stock_list 表读取全部（含 ST）")
    parser.add_argument("--layout", default=DEFAULT_LAYOUT, help=f"布局编号，默认 {DEFAULT_LAYOUT}")
    parser.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD（默认全历史）")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--cookie-file", default=DEFAULT_COOKIE_FILE, help="cookie 文件")
    parser.add_argument("--texts-file", default=DEFAULT_TEXTS_FILE, help="加密 text 缓存文件")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB 文件路径")
    parser.add_argument("--stage", default=None,
                        help="暂存 DuckDB 路径；auto 自动生成 download/temp/guzhi_stage_<时间戳>_<pid>.duckdb；不设则直入主库")
    parser.add_argument("--merge", default=None,
                        help="合并模式：暂存文件 glob，如 \"download/temp/guzhi_stage_*.duckdb\"，合并入 --db 后退出")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"每股间隔秒数，默认 {DEFAULT_DELAY}")
    parser.add_argument("--log", default=None,
                        help="日志文件路径；auto 自动生成 download/temp/guzhi_<时间戳>.log")
    parser.add_argument("--no-proxy", action="store_true", help="不走本机 7897 代理")
    parser.add_argument("--full", action="store_true",
                        help="全量重抓所有股票（覆盖已有数据）；默认增量，只抓未入库的股票")
    parser.add_argument("--no-refresh-texts", action="store_true",
                        help="抓取前不重新提取布局加密 text，直接用 texts-file 里的缓存")
    args = parser.parse_args()

    if args.merge:
        merge_stages(args.merge, db_path=args.db)
        return

    run(
        codes=args.codes,
        layout_id=args.layout,
        start=args.start,
        end=args.end,
        cookie_file=args.cookie_file,
        texts_file=args.texts_file,
        db_path=args.db,
        stage_path=args.stage,
        use_proxy=not args.no_proxy,
        delay=args.delay,
        log_path=args.log,
        full=args.full,
        refresh_texts=not args.no_refresh_texts,
    )


if __name__ == "__main__":
    main()
