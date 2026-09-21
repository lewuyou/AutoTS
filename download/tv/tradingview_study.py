# -*- coding: utf-8 -*-
"""TradingView 布局指标数据抓取入库模块（含收费/私有指标，纯 API）。

原理：
    网页端"下载图表数据"会把布局里的主品种 OHLC + 所有指标线导出成 CSV，
    指标数据是前端通过 WebSocket 的 create_study 拉的。create_study 的参数里
    脚本身份是端到端加密的（text 字段），但该加密 text 对同一指标长期固定、
    可复用（已验证跨会话一致、换 symbol 有效）。本模块复用事先抓好的 text，
    用布局 HTML 里提取的 auth_token 认证，换 symbol 批量拉指标数据。

依赖：
    pip install duckdb websocket-client requests

配置（一次性）：
    1. cookie：tv_cookie.txt（TradingView 登录 cookie，含 sessionid）
    2. 加密 text：tv_study_texts.json（从浏览器 WebSocket 帧抓的各指标 text）
    3. 布局配置：tv_layout_{布局编号}.json（布局 HTML 的 initData.content，含指标 inputs）

用法（独立运行）：
    python -m AutoTS.download.tv.tradingview_study                          # 默认布局+默认股票
    python -m AutoTS.download.tv.tradingview_study --codes 688223,300999
    python -m AutoTS.download.tv.tradingview_study --layout dP9MRLfC --start 2026-01-01
    python -m AutoTS.download.tv.tradingview_study --no-proxy

入库表 tradingview_study 字段含义（长表）：
    symbol      股票代码
    date        交易日
    study_id    布局里指标的 id（如 RB5CWA）
    metainfo    指标脚本标识（如 Script$STD;RSI@tv-scripting-101）
    study_name  可读指标名（如 RSI / GM_V2_KDJ，见 STUDY_NAMES）
    plot_idx    输出线序号（一个指标有多条输出线，如 MACD/Signal/Histogram）
    value       指标值（空值已过滤）

当前布局 dP9MRLfC 各指标 plot_idx 对照（名称已从布局图例确认）：
    RB5CWA  RSI (14, 收价, SMA 信号)
            plot0=RSI 值(0-100)  plot1=中线50(恒定)  plot2=RSI 的 SMA 信号线
            plot7/10/12/13=带宽/背景等辅助 plot（恒为 2 或 0，无分析价值）
    DBco3o  PEG比率 (Fund_price_earnings_growth_ratio)
            plot0=PEG 值
    KjCwTh  ADX and DI (14, 20)  —— 与 6Qwk52 为同一指标重复加载，数值一致
            plot0/plot1/plot2 = +DI / -DI / ADX（待对照确认顺序）
    6Qwk52  ADX and DI (14, 20)  同上
    LVUaCK  GM_V2_KDJ (9, 3)
            plot0/plot1/plot2 = K / D / J  plot3=0/1 信号
    DJtOcO  SQZMOM + time frame (日线, 20, 2)
            plot0=动量值  plot1=状态(0-3)  plot2=恒0  plot3=常量(4-6)
    fxx3m9  CM_Ult_MacD_MTF (60分/12/26/9)  多周期 MACD
            plot0/2/4/6=各周期 MACD 值  plot1/3/7=对应信号/档位(2-6 小整数)
    jT9ARG  TurnOver% 换手率 (20)
            plot0/plot1 = 换手率 / 其均线(或反之)
    SD8bVi  TMF资金流 (21)
            plot0=主值  plot1=正值部分  plot2=负值部分(0 填充)
    zKs55t  成交量流向 (130, 0.2, 2.5, 5)
            plot0=柱(恒0,可能未启用)  plot2/plot3=两条主线(数值接近)
    7eBG0W  OBV MACD Indicator (DEMA 9/26)
            plot0=恒0  plot1=OBV MACD 值  plot2=0/1 信号
"""

import argparse
import datetime
import json
import os
import random
import re
import string
import sys
import time

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    import websocket
except ImportError:
    websocket = None

try:
    import requests
except ImportError:
    requests = None

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "autots.duckdb")
DEFAULT_CODES = ["688223"]
DEFAULT_LAYOUT = "dP9MRLfC"
TABLE_NAME = "tradingview_study"
WS_URL = "wss://prodata.tradingview.com/socket.io/websocket?from=chart%2F{layout}%2F&date={date}&type=chart&auth=sessionid"
PAGE_URL = "https://cn.tradingview.com/chart/{layout}/"
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7897
N_BARS = 5000
EMPTY = 1e100  # TradingView 用 1e+100 表示空值
MAX_RETRY = 3      # 每个股票完整性抓取的最大轮数（首抓 + 对缺失指标重试）
RETRY_DELAY = 3    # 每轮重试间隔秒数

_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_COOKIE_FILE = os.path.join(_DIR, "tv_cookie.txt")
DEFAULT_TEXTS_FILE = os.path.join(_DIR, "tv_study_texts.json")

# 布局 dP9MRLfC 各 study_id 的可读指标名（从布局页面图例读取，用于 study_name 列）
STUDY_NAMES = {
    "7eBG0W": "OBV MACD Indicator",
    "fxx3m9": "CM_Ult_MacD_MTF",
    "LVUaCK": "GM_V2_KDJ",
    "KjCwTh": "ADX and DI",
    "6Qwk52": "ADX and DI",
    "RB5CWA": "RSI",
    "DJtOcO": "SQZMOM + time frame",
    "DBco3o": "PEG比率",
    "jT9ARG": "TurnOver% 换手率",
    "zKs55t": "成交量流向",
    "SD8bVi": "TMF资金流",
}


def _require():
    if duckdb is None:
        raise ImportError("缺少 duckdb，请安装：python3 -m pip install duckdb")
    if websocket is None:
        raise ImportError("缺少 websocket-client，请安装：python3 -m pip install websocket-client")
    if requests is None:
        raise ImportError("缺少 requests，请安装：python3 -m pip install requests")


def code_to_symbol(code):
    """A 股代码 -> TradingView symbol。"""
    code = str(code).strip()
    if ":" in code:
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


def _frame(func, params):
    body = json.dumps({"m": func, "p": params}, separators=(",", ":"))
    return "~m~" + str(len(body)) + "~m~" + body


def _proxies(use_proxy):
    if not use_proxy:
        return None
    p = f"http://{PROXY_HOST}:{PROXY_PORT}"
    return {"http": p, "https": p}


def fetch_layout_page(layout_id, cookie, use_proxy=True):
    """带 cookie 请求布局 HTML，提取 auth_token 和 initData.content（指标配置）。"""
    url = PAGE_URL.format(layout=layout_id)
    r = requests.get(url, headers={"Cookie": cookie, "User-Agent": "Mozilla/5.0"},
                     proxies=_proxies(use_proxy), timeout=30)
    html = r.text
    m = re.search(r'"auth_token":"(eyJ[^"]+)"', html)
    if not m:
        raise RuntimeError("布局页面未提取到 auth_token，cookie 可能已失效")
    auth_token = m.group(1)

    # 提取 initData.content（平衡大括号）
    i = html.find("initData.content = ")
    content = None
    if i >= 0:
        s = html[i + len("initData.content = "):]
        depth = 0
        end = 0
        for k, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = k + 1
                    break
        if end:
            content = json.loads(s[:end])
    return auth_token, content


def load_layout_config(layout_id, cookie, use_proxy=True, cache=True):
    """加载布局配置：优先读缓存文件，否则请求布局 HTML 解析并缓存。"""
    cache_file = os.path.join(_DIR, f"tv_layout_{layout_id}.json")
    if cache and os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            content = json.load(f)
        # 仍需 auth_token（每次从页面取，会过期）
        auth_token, _ = fetch_layout_page(layout_id, cookie, use_proxy)
        return auth_token, content
    auth_token, content = fetch_layout_page(layout_id, cookie, use_proxy)
    if content is None:
        raise RuntimeError("布局页面未解析到 initData.content")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)
    return auth_token, content


def parse_studies(content, texts):
    """从布局配置解析 study 列表：[{id, metainfo, inputs(原始), text}]，跳过无加密 text 的。"""
    studies = []
    charts = content.get("charts", [])
    if not charts:
        return studies
    for pane in charts[0].get("panes", []):
        for s in pane.get("sources", []):
            if s.get("type") == "MainSeries":
                continue
            sid = s.get("id")
            metainfo = s.get("metaInfo", "")
            inputs = s.get("state", {}).get("inputs", {})
            info = texts.get(sid)
            if not info or "text" not in info:
                continue  # 没有加密 text 的指标跳过（如分红/拆股标记）
            studies.append({
                "id": sid,
                "metainfo": metainfo,
                "inputs": inputs,
                "text": info["text"],
                "pineId": info.get("pineId", ""),
                "pineVersion": info.get("pineVersion", ""),
            })
    return studies


def _wrap_value(v):
    if isinstance(v, bool):
        t = "bool"
    elif isinstance(v, int):
        t = "integer"
    elif isinstance(v, float):
        t = "float"
    else:
        t = "text"
    return {"v": v, "f": True, "t": t}


def build_study_param(study):
    """构造 create_study 的第6个参数：text + pineId/pineVersion + 明文 inputs。"""
    param = {"text": study["text"]}
    if study.get("pineId"):
        param["pineId"] = study["pineId"]
    if study.get("pineVersion"):
        param["pineVersion"] = study["pineVersion"]
    for k, v in study["inputs"].items():
        if k in ("pineId", "pineVersion", "__user_pro_plan"):
            continue
        if k == "pineFeatures":
            param[k] = {"v": v if isinstance(v, str) else json.dumps(v), "f": True, "t": "text"}
            continue
        param[k] = _wrap_value(v)
    return param


def fetch_symbol_studies(symbol, studies, auth_token, layout_id, cookie, use_proxy=True, timeout=40):
    """连 prodata WebSocket，对一个 symbol 拉指定 study 列表的数据。

    返回 (数据 dict {study_id: [(ts, [v...])]}, 缺失的 study_id 列表)。
    "缺失" = 该 study 既没收到 study_completed 也没拿到任何数据点。
    """
    ws_url = WS_URL.format(layout=layout_id,
                           date=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%%3A%M%%3A%S"))
    headers = ["Cookie: " + cookie]
    kwargs = dict(timeout=30, origin="https://cn.tradingview.com")
    if use_proxy:
        kwargs.update(http_proxy_host=PROXY_HOST, http_proxy_port=PROXY_PORT, proxy_type="http")
    ws = websocket.create_connection(ws_url, header=headers, **kwargs)

    qs = _gen_session("qs")
    cs = _gen_session("cs")
    try:
        ws.send(_frame("set_auth_token", [auth_token]))
        ws.send(_frame("set_locale", ["zh-Hans", "CN"]))
        ws.send(_frame("chart_create_session", [cs, ""]))
        ws.send(_frame("switch_timezone", [cs, "Asia/Hong_Kong"]))
        ws.send(_frame("quote_create_session", [qs]))
        ws.send(_frame("resolve_symbol", [cs, "symbol_1",
                                          '={"adjustment":"splits","currency-id":"CNY","symbol":"' + symbol + '"}']))
        ws.send(_frame("create_series", [cs, "s1", "s1", "symbol_1", "1D", N_BARS]))
        for st in studies:
            ws.send(_frame("create_study", [cs, st["id"], "st1", "s1",
                                            "Script@tv-scripting-101!", build_study_param(st)]))

        raw = ""
        ws.settimeout(timeout)
        done = set()
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                chunk = ws.recv()
            except Exception:
                break
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", "ignore")
            raw += chunk
            for mm in re.findall(r"~m~\d+~m~(\{.*?\})(?=~m~|$)", chunk, re.S):
                try:
                    o = json.loads(mm)
                except Exception:
                    continue
                if o.get("m") == "study_completed":
                    p = o.get("p", [])
                    if len(p) > 1:
                        done.add(p[1])
            if len(done) >= len(studies):
                break
    finally:
        ws.close()

    if "symbol_error" in raw:
        raise RuntimeError(f"{symbol}: 品种解析失败")

    # 解析 du 消息里每个 study 的数据
    out = {}
    for mm in re.findall(r"~m~\d+~m~(\{.*?\})(?=~m~|$)", raw, re.S):
        try:
            o = json.loads(mm)
        except Exception:
            continue
        if o.get("m") != "du":
            continue
        p1 = o.get("p", [None, {}])[1]
        if not isinstance(p1, dict):
            continue
        for st_key, st_val in p1.items():
            if not isinstance(st_val, dict) or "st" not in st_val:
                continue
            pts = []
            for it in st_val.get("st", []):
                v = it.get("v")
                if v:
                    pts.append((v[0], v[1:]))
            if pts:
                out.setdefault(st_key, []).extend(pts)

    # 完整性校验：有数据点的 study 才算成功。
    # 基本面指标（Internal$STD;Fund_ 前缀）数据稀疏，可能本就无数据，不参与完整性校验。
    expected = {st["id"] for st in studies
                if not st.get("metainfo", "").startswith("Internal$STD;Fund_")}
    got = {sid for sid, pts in out.items() if pts}
    missing = sorted(expected - got)
    return out, missing


def ts_to_date(ts):
    """UTC 时间戳 -> 北京时间交易日 YYYY-MM-DD。

    TradingView 日频 K 线时间戳 = 该市场开盘时刻的 UTC 时间，
    全球主要市场开盘时刻换算成 UTC+8 后都落在同一天，故统一按 UTC+8 转日期即可。
    """
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc) \
        .astimezone(datetime.timezone(datetime.timedelta(hours=8))).date().isoformat()


def studies_to_rows(symbol_code, study_data, studies, start=None, end=None):
    """把 {study_id: [(ts, [v...])]} 展平成长表行，过滤空值。"""
    meta_map = {st["id"]: st["metainfo"] for st in studies}
    rows = []
    for st_id, pts in study_data.items():
        metainfo = meta_map.get(st_id, "")
        study_name = STUDY_NAMES.get(st_id, "")
        for ts, vals in pts:
            d = ts_to_date(ts)
            if start and d < start:
                continue
            if end and d > end:
                continue
            for idx, val in enumerate(vals):
                if val is None or val == EMPTY or (isinstance(val, float) and val >= EMPTY):
                    continue
                if isinstance(val, bool):
                    continue
                rows.append((symbol_code, d, st_id, metainfo, study_name, idx, float(val)))
    return rows


def ingest(rows, db_path=DEFAULT_DB_PATH):
    _require()
    con = duckdb.connect(db_path)
    try:
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                symbol TEXT,
                date DATE,
                study_id TEXT,
                metainfo TEXT,
                study_name TEXT,
                plot_idx INTEGER,
                value DOUBLE,
                PRIMARY KEY (symbol, date, study_id, plot_idx)
            )
            """
        )
        # 兼容旧表（无 study_name 列）
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()]
        if "study_name" not in cols:
            con.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN study_name TEXT")
        con.executemany(
            f"INSERT OR REPLACE INTO {TABLE_NAME} (symbol, date, study_id, metainfo, study_name, plot_idx, value) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    finally:
        con.close()
    return len(rows)


def run(codes=None, layout_id=DEFAULT_LAYOUT, start=None, end=None,
        cookie_file=None, texts_file=None, db_path=None, use_proxy=True):
    """供外部调用的入口。

    参数：
        codes: 股票代码列表，默认 ["688223"]
        layout_id: 布局编号，默认 dP9MRLfC
        start/end: 日期过滤 YYYY-MM-DD
        cookie_file: cookie 文件，默认 download/tv/tv_cookie.txt
        texts_file: 加密 text 缓存，默认 download/tv/tv_study_texts.json
        db_path: DuckDB 路径，默认 download/autots.duckdb
        use_proxy: 是否走本机 7897 代理
    返回：
        dict 含 rows_count, db_path, series 等
    """
    _require()
    if codes is None:
        codes = DEFAULT_CODES
    if isinstance(codes, str):
        codes = [c.strip() for c in codes.split(",") if c.strip()]
    if not codes:
        raise ValueError("未提供股票代码")

    cookie_file = cookie_file or DEFAULT_COOKIE_FILE
    texts_file = texts_file or DEFAULT_TEXTS_FILE
    with open(cookie_file, encoding="utf-8") as f:
        cookie = f.read().strip()
    with open(texts_file, encoding="utf-8") as f:
        texts = json.load(f)

    end = end or datetime.date.today().isoformat()

    print(f"[TradingView指标] 布局 {layout_id}，提取 auth_token + 指标配置 ...")
    auth_token, content = load_layout_config(layout_id, cookie, use_proxy=use_proxy)
    studies = parse_studies(content, texts)
    if not studies:
        raise RuntimeError("布局里没有可抓取的指标（缺加密 text 或无 study）")
    print(f"[TradingView指标] 布局含 {len(studies)} 个可抓取指标")

    db_path = db_path or DEFAULT_DB_PATH
    total_rows = 0
    series = {}
    failed = {}
    incomplete = {}

    for code in codes:
        symbol = code_to_symbol(code)
        print(f"[TradingView指标] 抓取 {code} ({symbol}) ...")

        # 完整性抓取：首抓全部，对缺失指标重试（最多 MAX_RETRY 轮），合并数据
        study_data = {}
        pending = list(studies)
        last_err = None
        for attempt in range(1, MAX_RETRY + 1):
            try:
                data, missing = fetch_symbol_studies(symbol, pending, auth_token, layout_id,
                                                     cookie, use_proxy=use_proxy)
            except Exception as e:
                last_err = e
                print(f"[TradingView指标] {code} 第{attempt}轮连接失败: {e}", file=sys.stderr)
                time.sleep(RETRY_DELAY)
                continue
            # 合并本轮数据
            for sid, pts in data.items():
                study_data.setdefault(sid, []).extend(pts)
            if not missing:
                break
            # 有缺失：只针对缺失指标重试
            if attempt < MAX_RETRY:
                miss_names = [m for m in missing]
                print(f"[TradingView指标] {code} 第{attempt}轮缺 {len(missing)} 个指标 {miss_names}，{RETRY_DELAY}s 后重试 ...")
                pending = [st for st in studies if st["id"] in missing]
                time.sleep(RETRY_DELAY)
            else:
                incomplete[code] = missing
                print(f"[TradingView指标] {code} 重试 {MAX_RETRY} 轮后仍缺 {len(missing)} 个指标: {missing}", file=sys.stderr)

        if not study_data:
            failed[code] = str(last_err) if last_err else "无数据"
            print(f"[TradingView指标] {code} 失败: {failed[code]}", file=sys.stderr)
            continue

        rows = studies_to_rows(code, study_data, studies, start=start, end=end)
        if not rows:
            failed[code] = "无数据"
            print(f"[TradingView指标] {code} 无数据", file=sys.stderr)
            continue
        n = ingest(rows, db_path=db_path)
        total_rows += n
        dates = [r[1] for r in rows]
        n_studies = len(set(r[2] for r in rows))
        series[code] = {"n": len(rows), "studies": n_studies, "first": min(dates), "last": max(dates)}
        mark = "" if code not in incomplete else f"  [缺{len(incomplete[code])}指标]"
        print(f"[TradingView指标] {code} 入库 {n} 行  {n_studies} 个指标  {min(dates)} ~ {max(dates)}{mark}")

    print("\n[TradingView指标] 入库概览")
    for code in sorted(series):
        s = series[code]
        mark = "" if code not in incomplete else f"  [缺指标: {incomplete[code]}]"
        print(f"  {code}: n={s['n']:6d}  studies={s['studies']}  {s['first']} ~ {s['last']}{mark}")

    result = {"rows_count": total_rows, "db_path": db_path, "series": series}
    if incomplete:
        result["incomplete"] = incomplete
    if failed:
        result["failed"] = failed
        if not series:
            raise RuntimeError(f"全部失败: {failed}")
    return result


def main():
    parser = argparse.ArgumentParser(description="TradingView 布局指标抓取入库（含收费/私有指标）")
    parser.add_argument("--codes", default=",".join(DEFAULT_CODES), help="逗号分隔股票代码")
    parser.add_argument("--layout", default=DEFAULT_LAYOUT, help="布局编号，默认 dP9MRLfC")
    parser.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--cookie-file", default=DEFAULT_COOKIE_FILE, help="cookie 文件")
    parser.add_argument("--texts-file", default=DEFAULT_TEXTS_FILE, help="加密 text 缓存文件")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB 文件路径")
    parser.add_argument("--no-proxy", action="store_true", help="不走本机 7897 代理")
    args = parser.parse_args()

    run(
        codes=args.codes,
        layout_id=args.layout,
        start=args.start,
        end=args.end,
        cookie_file=args.cookie_file,
        texts_file=args.texts_file,
        db_path=args.db,
        use_proxy=not args.no_proxy,
    )


if __name__ == "__main__":
    main()
