# -*- coding: utf-8 -*-
"""TradingView 日频 K 线抓取入库模块（OHLC + 成交量）。

通过 TradingView 历史数据 WebSocket 协议（wss://data.tradingview.com）拉取，
对应网页端"下载图表数据"导出的 CSV（time,open,high,low,close），并额外带上成交量。
公开日频行情无需登录即可拉取；保留 cookie 支持以兼容需登录的品种。

可独立运行，也可由 download_all.py 调用 run()。

用法（独立运行）：
    python -m AutoTS.download.tradingview                      # 抓默认股票，全历史
    python -m AutoTS.download.tradingview --codes 688223,300999
    python -m AutoTS.download.tradingview --start 2023-01-01   # 只入库该日期之后
    python -m AutoTS.download.tradingview --cookie-file c.txt  # 从文件读 Cookie（可选）
    python -m AutoTS.download.tradingview --no-proxy           # 不走代理

依赖：
    pip install duckdb websocket-client

入库表 tradingview 字段含义：
    symbol  股票代码（如 "688223"，自动映射上交所 SSE；6 开头->SSE，0/3 开头->SZSE）
    date    交易日（由 UTC 时间戳转北京时间日期）
    open/high/low/close  开高低收
    volume  成交量（股）
"""

import argparse
import datetime
import json
import os
import random
import re
import string
import sys

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    import websocket
except ImportError:
    websocket = None

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autots.duckdb")
DEFAULT_CODES = ["688223"]
TABLE_NAME = "tradingview"
WS_URL = "wss://data.tradingview.com/socket.io/websocket"
N_BARS = 5000  # 单次拉取的 K 线根数上限（覆盖日频全历史）
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7897


def _require_duckdb():
    if duckdb is None:
        raise ImportError("缺少 duckdb，请安装：python3 -m pip install duckdb")


def _require_websocket():
    if websocket is None:
        raise ImportError("缺少 websocket-client，请安装：python3 -m pip install websocket-client")


def parse_cookie_text(text):
    """把用户粘贴的 Cookie 解析成 {name: value}，兼容 JSON 列表与 k=v; 串。"""
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return {c["name"]: c["value"] for c in data if "name" in c and "value" in c}
        except Exception:
            pass
    pairs = {}
    for part in text.replace(";", "\n").split("\n"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        if k and k.lower() != "undefined":
            pairs[k] = v.strip()
    return pairs


def code_to_symbol(code):
    """A 股代码 -> TradingView symbol。6/9 开头上交所，0/2/3 开头深交所，4/8 开头北交所。"""
    code = str(code).strip()
    if ":" in code:  # 已是 EXCHANGE:CODE 形式
        return code
    if code[0] in ("6", "9"):
        return f"SSE:{code}"
    if code[0] in ("0", "2", "3"):
        return f"SZSE:{code}"
    if code[0] in ("4", "8"):
        return f"BSE:{code}"
    return f"SSE:{code}"


def _gen_session(prefix):
    return prefix + "_" + "".join(random.choice(string.ascii_lowercase) for _ in range(12))


def _frame(payload_str):
    return "~m~" + str(len(payload_str)) + "~m~" + payload_str


def _send(ws, func, args):
    ws.send(_frame(json.dumps({"m": func, "p": args}, separators=(",", ":"))))


def fetch_bars(symbol, interval="1D", n_bars=N_BARS, cookie=None, use_proxy=True, timeout=30):
    """连接 WebSocket 拉取一个品种的 K 线，返回 [[ts, o, h, l, c, v], ...]（按时间升序）。"""
    _require_websocket()
    headers = []
    if cookie:
        headers.append("Cookie: " + cookie)
    kwargs = dict(timeout=timeout, origin="https://cn.tradingview.com")
    if use_proxy:
        kwargs.update(http_proxy_host=PROXY_HOST, http_proxy_port=PROXY_PORT, proxy_type="http")
    ws = websocket.create_connection(WS_URL, header=headers, **kwargs)

    qs = _gen_session("qs")
    cs = _gen_session("cs")
    try:
        _send(ws, "set_auth_token", ["unauthorized_user_token"])
        _send(ws, "chart_create_session", [cs, ""])
        _send(ws, "quote_create_session", [qs])
        _send(ws, "quote_set_fields", [qs, "ch", "chp", "current_session", "description",
                                       "local_description", "language", "exchange", "fractional",
                                       "is_tradable", "lp", "lp_time", "minmov", "minmove2",
                                       "original_name", "pricescale", "pro_name", "short_name",
                                       "type", "update_mode", "volume", "currency_code", "rchp", "rtc"])
        _send(ws, "quote_add_symbols", [qs, symbol, {"flags": ["force_permission"]}])
        _send(ws, "quote_fast_symbols", [qs, symbol])
        _send(ws, "resolve_symbol", [cs, "symbol_1",
                                     '={"symbol":"' + symbol + '","adjustment":"splits","session":"extended"}'])
        _send(ws, "create_series", [cs, "s1", "s1", "symbol_1", interval, n_bars])

        raw = ""
        ws.settimeout(timeout)
        while True:
            chunk = ws.recv()
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", "ignore")
            raw += chunk
            if "series_completed" in chunk:
                break
    finally:
        ws.close()

    if '"m":"symbol_error"' in raw.replace(" ", "") or '"m": "symbol_error"' in raw:
        raise RuntimeError(f"{symbol}: 品种解析失败")

    bars = []
    for m in re.findall(r"~m~\d+~m~(\{.*?\})(?=~m~|$)", raw, re.S):
        try:
            obj = json.loads(m)
        except Exception:
            continue
        if obj.get("m") == "timescale_update":
            s1 = obj.get("p", [{}, {}])[1].get("s1", {})
            for it in s1.get("s", []):
                v = it.get("v")
                if v and len(v) >= 5:
                    vol = v[5] if len(v) > 5 else None
                    bars.append([v[0], v[1], v[2], v[3], v[4], vol])
    bars.sort(key=lambda b: b[0])
    if not bars:
        raise RuntimeError(f"{symbol}: 未返回 K 线数据")
    return bars


def ts_to_date(ts):
    """UTC 时间戳 -> 北京时间交易日 YYYY-MM-DD。"""
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc) \
        .astimezone(datetime.timezone(datetime.timedelta(hours=8))).date().isoformat()


def bars_to_rows(code, bars, start=None, end=None):
    """转成 (symbol, date, o, h, l, c, volume) 行，按 start/end 过滤。"""
    rows = []
    for b in bars:
        d = ts_to_date(b[0])
        if start and d < start:
            continue
        if end and d > end:
            continue
        rows.append((code, d, b[1], b[2], b[3], b[4], b[5]))
    return rows


def ingest(rows, db_path=DEFAULT_DB_PATH):
    """UPSERT 到 DuckDB 的 tradingview 表。"""
    _require_duckdb()
    con = duckdb.connect(db_path)
    try:
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                symbol TEXT,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (symbol, date)
            )
            """
        )
        con.executemany(
            f"INSERT OR REPLACE INTO {TABLE_NAME} (symbol, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    finally:
        con.close()
    return len(rows)


def write_csv(rows, base_dir, code):
    """按股票代码写 CSV（与网页导出格式一致，外加 volume 列）。"""
    path = os.path.join(base_dir, f"tradingview_{code}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        f.write("time,open,high,low,close,volume\n")
        for _, d, o, h, l, c, v in rows:
            f.write(f"{d},{o},{h},{l},{c},{'' if v is None else v}\n")
    return path


def run(codes=None, start=None, end=None, cookie_file=None, db_path=None,
        interval="1D", use_proxy=True, no_csv=False, csv_dir=None):
    """供外部调用的入口。

    参数：
        codes: 股票代码列表，默认 ["688223"]
        start: 起始日期 YYYY-MM-DD，None 则入库全部历史
        end: 结束日期 YYYY-MM-DD，默认今天
        cookie_file: Cookie 文件路径（可选，公开日频无需登录）
        db_path: DuckDB 文件路径，默认模块目录下 autots.duckdb
        interval: K 线周期，默认 1D
        use_proxy: 是否走本机 7897 代理（默认 True）
        no_csv: 是否跳过 CSV 导出
        csv_dir: CSV 输出目录，默认模块目录下 temp/
    返回：
        dict 包含 rows_count, db_path, csv_paths, series 等
    """
    if codes is None:
        codes = DEFAULT_CODES
    if isinstance(codes, str):
        codes = [c.strip() for c in codes.split(",") if c.strip()]
    if not codes:
        raise ValueError("未提供股票代码")

    today = datetime.date.today().isoformat()
    end = end or today

    cookie = None
    if cookie_file:
        with open(cookie_file, encoding="utf-8") as f:
            pairs = parse_cookie_text(f.read())
        if pairs:
            cookie = "; ".join(f"{k}={v}" for k, v in pairs.items())

    db_path = db_path or DEFAULT_DB_PATH
    csv_dir = csv_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    if not no_csv:
        os.makedirs(csv_dir, exist_ok=True)

    total_rows = 0
    csv_paths = {}
    series = {}
    failed = {}

    for code in codes:
        symbol = code_to_symbol(code)
        print(f"[TradingView] 抓取 {code} ({symbol}) {interval} ...")
        try:
            bars = fetch_bars(symbol, interval=interval, cookie=cookie, use_proxy=use_proxy)
        except Exception as e:
            failed[code] = str(e)
            print(f"[TradingView] {code} 失败: {e}", file=sys.stderr)
            continue
        rows = bars_to_rows(code, bars, start=start, end=end)
        if not rows:
            failed[code] = "无数据"
            print(f"[TradingView] {code} 无数据", file=sys.stderr)
            continue
        n = ingest(rows, db_path=db_path)
        total_rows += n
        if not no_csv:
            csv_paths[code] = write_csv(rows, csv_dir, code)
        closes = [r[5] for r in rows if r[5] is not None]
        series[code] = {
            "n": len(rows),
            "first": rows[0][1],
            "last": rows[-1][1],
            "mean_close": sum(closes) / len(closes) if closes else 0,
        }
        print(f"[TradingView] {code} 入库 {n} 行  {rows[0][1]} ~ {rows[-1][1]}")

    print("\n[TradingView] 入库概览")
    for code in sorted(series):
        s = series[code]
        print(f"  {code}: n={s['n']:5d}  mean_close={s['mean_close']:10.2f}  {s['first']} ~ {s['last']}")

    result = {
        "rows_count": total_rows,
        "db_path": db_path,
        "csv_paths": csv_paths or None,
        "series": series,
    }
    if failed:
        result["failed"] = failed
        if not series:
            raise RuntimeError(f"全部失败: {failed}")
    return result


def main():
    parser = argparse.ArgumentParser(description="TradingView 日频 K 线抓取入库（OHLC + 成交量）")
    parser.add_argument("--codes", default=",".join(DEFAULT_CODES), help="逗号分隔股票代码，如 688223,300999")
    parser.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD（不传则入库全部历史）")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--interval", default="1D", help="K 线周期，默认 1D")
    parser.add_argument("--cookie-file", default=None, help="从文件读取 Cookie（可选）")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB 文件路径")
    parser.add_argument("--no-proxy", action="store_true", help="不走本机 7897 代理")
    parser.add_argument("--no-csv", action="store_true", help="不写 CSV，只入库")
    args = parser.parse_args()

    run(
        codes=args.codes,
        start=args.start,
        end=args.end,
        cookie_file=args.cookie_file,
        db_path=args.db,
        interval=args.interval,
        use_proxy=not args.no_proxy,
        no_csv=args.no_csv,
    )


if __name__ == "__main__":
    main()
