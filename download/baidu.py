# -*- coding: utf-8 -*-
"""百度指数数据抓取入库模块（搜索指数 search_all + 资讯指数 feed，日频）。

可独立运行，也可由 download_all.py 调用 run()。

用法（独立运行）：
    python -m AutoTS.download.baidu                       # 交互式粘贴 Cookie，抓当年日频
    python -m AutoTS.download.baidu --cookie-file c.txt   # 从文件读 Cookie
    python -m AutoTS.download.baidu --keywords 金龙鱼,浪潮信息
    python -m AutoTS.download.baidu --start 2013-01-01    # 抓指定窗口日频
    python -m AutoTS.download.baidu --headless            # 无头模式

依赖：
    pip install duckdb playwright
    playwright install chromium

入库表 baidu 字段含义：
    keyword  搜索关键词（如 "金龙鱼"）
    source   指数类型：search_all=搜索指数（整体=PC+移动），feed=资讯指数
    date     日期（日频）
    value    指数值（整数，空值记 None；资讯指数自 2017-07-03 才有数据）
"""

import argparse
import csv
import datetime
import json
import os
import sys

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autots.duckdb")
DEFAULT_KEYWORDS = ["金龙鱼"]
TABLE_NAME = "baidu"
SEARCH_START = "2011-01-01"   # search_all（整体=PC+移动）最早日期
FEED_START = "2017-07-03"     # 资讯指数最早日期
CHUNK_DAYS = 360              # 日频单次请求最大天数（实测 365 内返回日频），留余量

# 页面内执行：取 token → 分段抓 search_all / feed → ptbk 解码 → 返回 {search:{kw:[chunk]}, feed:{kw:[chunk]}}
# arg = {keywords: [...], searchChunks: [[start,end],...], feedChunks: [[start,end],...]}
SCRAPE_JS = r"""
async (arg) => {
  const keywords = arg.keywords;
  const searchChunks = arg.searchChunks || [];
  const feedChunks = arg.feedChunks || [];

  function decrypt(ptbk, data) {
    if (!data) return '';
    const n = ptbk.split(''), i = data.split(''), a = {}, r = [];
    for (let o = 0; o < n.length / 2; o++) a[n[o]] = n[n.length / 2 + o];
    for (let s = 0; s < i.length; s++) r.push(a[i[s]]);
    return r.join('');
  }

  async function getToken(retries = 20) {
    for (let i = 0; i < retries; i++) {
      try {
        const t = await new Promise((resolve, reject) => {
          window.Paris.getAcsInstance((err, inst) => {
            if (err) return reject(String(err));
            inst.getSign((e, s) => (e ? reject(String(e)) : resolve(s)));
          });
        });
        if (t && t !== 'NONE') return t;
      } catch (e) { /* SDK 尚未就绪，重试 */ }
      await new Promise(r => setTimeout(r, 500));
    }
    throw new Error('无法生成 Cipher-Text token（Paris SDK 未就绪）');
  }

  const token = await getToken();
  const H = { credentials: 'include', headers: { 'Cipher-Text': token, 'Accept': 'application/json, text/plain, */*' } };
  const word = JSON.stringify(keywords.map(k => [{ name: k, wordType: 1 }]));
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  async function getPtbk(uniqid) {
    const r = await (await fetch('https://index.baidu.com/Interface/ptbk?uniqid=' + uniqid, H)).json();
    if (r.status !== 0 || !r.data) throw new Error('ptbk status=' + r.status);
    return r.data;
  }

  const searchOut = {};
  for (const [s, e] of searchChunks) {
    const qs = new URLSearchParams({ area: '0', word, startDate: s, endDate: e });
    const r = await (await fetch('https://index.baidu.com/api/SearchApi/index?' + qs.toString(), H)).json();
    if (r.status !== 0) throw new Error('SearchApi status=' + r.status + ' ' + (r.message || '') + ' chunk=' + s + '~' + e);
    const ptbk = await getPtbk(r.data.uniqid);
    for (const ui of (r.data.userIndexes || [])) {
      const kw = (ui.word || []).map(x => x.name).join('');
      (searchOut[kw] = searchOut[kw] || []).push({
        start: ui.all.startDate, end: ui.all.endDate,
        values: decrypt(ptbk, ui.all.data).split(','),
      });
    }
    await sleep(400);
  }

  const feedOut = {};
  for (const [s, e] of feedChunks) {
    const qs = new URLSearchParams({ area: '0', word, startDate: s, endDate: e });
    const r = await (await fetch('https://index.baidu.com/api/FeedSearchApi/getFeedIndex?' + qs.toString(), H)).json();
    if (r.status !== 0) throw new Error('FeedSearchApi status=' + r.status + ' ' + (r.message || '') + ' chunk=' + s + '~' + e);
    const ptbk = await getPtbk(r.data.uniqid);
    for (const it of (r.data.index || [])) {
      const kw = (it.key || []).map(x => x.name).join('');
      (feedOut[kw] = feedOut[kw] || []).push({
        start: it.startDate, end: it.endDate,
        values: decrypt(ptbk, it.data).split(','),
      });
    }
    await sleep(400);
  }

  return { search: searchOut, feed: feedOut };
}
"""


def _require_duckdb():
    if duckdb is None:
        raise ImportError("缺少 duckdb，请安装：python3 -m pip install duckdb")


def _require_playwright():
    if sync_playwright is None:
        raise ImportError("缺少 playwright，请安装：python3 -m pip install playwright && playwright install chromium")


def parse_cookie_text(text):
    """把用户粘贴的 Cookie 解析成 {name: value}，兼容多种格式。"""
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


def read_cookie_interactive():
    print("请粘贴浏览器抓取的 Cookie（在 index.baidu.com 登录后，DevTools 里复制请求头里的 Cookie 整串）：")
    print("粘贴后回车，再输入一个空行结束（直接 Ctrl+D 也可结束）。")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            if lines:
                break
            continue
        lines.append(line.strip())
    return " ".join(lines)


def build_chunks(start, end, chunk_days=CHUNK_DAYS):
    """把 [start, end] 切成连续的小段，每段跨度 <= chunk_days。"""
    s = datetime.date.fromisoformat(start)
    e = datetime.date.fromisoformat(end)
    chunks = []
    while s <= e:
        c_end = min(s + datetime.timedelta(days=chunk_days), e)
        chunks.append([s.isoformat(), c_end.isoformat()])
        s = c_end + datetime.timedelta(days=1)
    return chunks


def scrape(keywords, cookie_text, search_chunks, feed_chunks, headless=False):
    """启动浏览器注入 Cookie，分段抓取并解码，返回 payload。"""
    _require_playwright()
    pairs = parse_cookie_text(cookie_text)
    if not pairs:
        raise ValueError("未解析到任何 Cookie，请检查粘贴内容（应包含 BDUSS 等）。")
    cookies = [{"name": k, "value": v, "domain": ".baidu.com", "path": "/"} for k, v in pairs.items()]

    arg = {"keywords": keywords, "searchChunks": search_chunks, "feedChunks": feed_chunks}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        context.add_cookies(cookies)
        page = context.new_page()
        page.goto("https://index.baidu.com/v2/main/index.html", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_function("typeof window.Paris !== 'undefined'", timeout=30000)
        try:
            page.wait_for_load_state("networkidle", timeout=20000)
        except Exception:
            pass
        payload = page.evaluate(SCRAPE_JS, arg)
        browser.close()
    return payload


def daily_dates(start, end):
    """逐日日期序列（含两端）。"""
    s = datetime.date.fromisoformat(start)
    e = datetime.date.fromisoformat(end)
    out = []
    d = s
    while d <= e:
        out.append(d.isoformat())
        d += datetime.timedelta(days=1)
    return out


def payload_to_rows(payload):
    """展平成 (keyword, source, date, value) 行，空值记 None。"""
    rows = []
    for kw, chunks in payload.get("search", {}).items():
        for c in chunks:
            dates = daily_dates(c["start"], c["end"])
            for d, v in zip(dates, c["values"]):
                rows.append((kw, "search_all", d, None if v == "" else float(v)))
    for kw, chunks in payload.get("feed", {}).items():
        for c in chunks:
            dates = daily_dates(c["start"], c["end"])
            for d, v in zip(dates, c["values"]):
                rows.append((kw, "feed", d, None if v == "" else float(v)))
    return rows


def ingest(rows, db_path=DEFAULT_DB_PATH):
    """UPSERT 到 DuckDB 的 baidu 表。"""
    _require_duckdb()
    con = duckdb.connect(db_path)
    try:
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                keyword TEXT,
                source TEXT,
                date DATE,
                value DOUBLE,
                PRIMARY KEY (keyword, source, date)
            )
            """
        )
        con.executemany(
            f"INSERT OR REPLACE INTO {TABLE_NAME} (keyword, source, date, value) VALUES (?, ?, ?, ?)",
            rows,
        )
    finally:
        con.close()
    return len(rows)


def write_csv(rows, base_dir):
    """写长表 + 宽表 CSV，返回两个文件路径。"""
    long_path = os.path.join(base_dir, "baidu_index_long.csv")
    wide_path = os.path.join(base_dir, "baidu_index_wide.csv")

    with open(long_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["keyword", "source", "date", "value"])
        w.writerows(rows)

    wide = {}
    dates = set()
    for kw, src, d, v in rows:
        wide.setdefault(f"{kw}_{src}", {})[d] = v
        dates.add(d)
    colnames = sorted(wide)
    dates_sorted = sorted(dates)
    with open(wide_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date"] + colnames)
        for d in dates_sorted:
            w.writerow([d] + [wide[c].get(d, "") for c in colnames])
    return long_path, wide_path


def run(keywords=None, start=None, end=None, cookie_file=None, db_path=None, headless=False, no_csv=False, csv_dir=None):
    """供外部调用的入口。

    参数：
        keywords: 关键词列表，默认 ["金龙鱼"]
        start: 起始日期 YYYY-MM-DD，默认当年 1 月 1 日
        end: 结束日期 YYYY-MM-DD，默认今天
        cookie_file: Cookie 文件路径，None 则交互式输入
        db_path: DuckDB 文件路径，默认模块目录下 baidu_index.duckdb
        headless: 是否无头模式
        no_csv: 是否跳过 CSV 导出
        csv_dir: CSV 输出目录，默认模块目录下 temp/
    返回：
        dict 包含 rows_count, db_path, csv_paths 等
    """
    if keywords is None:
        keywords = DEFAULT_KEYWORDS
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    if not keywords:
        raise ValueError("未提供关键词")

    today = datetime.date.today()
    if start or end:
        end_d = datetime.date.fromisoformat(end) if end else today
        start_d = datetime.date.fromisoformat(start) if start else end_d - datetime.timedelta(days=30)
    else:
        end_d = today
        start_d = datetime.date(today.year, 1, 1)
    end_s = end_d.isoformat()
    start_s = start_d.isoformat()

    search_chunks = build_chunks(start_s, end_s)
    feed_start = max(start_d, datetime.date.fromisoformat(FEED_START))
    feed_chunks = build_chunks(feed_start.isoformat(), end_s) if feed_start <= end_d else []

    if cookie_file:
        with open(cookie_file, encoding="utf-8") as f:
            cookie_text = f.read()
    else:
        cookie_text = read_cookie_interactive()

    print(f"[百度指数] 关键词：{keywords}，search: {start_s} ~ {end_s}，feed: {feed_start.isoformat()} ~ {end_s}")
    payload = scrape(keywords, cookie_text, search_chunks, feed_chunks, headless=headless)

    rows = payload_to_rows(payload)
    db_path = db_path or DEFAULT_DB_PATH
    n = ingest(rows, db_path=db_path)
    print(f"[百度指数] 入库 {n} 行 -> {db_path}")

    csv_paths = None
    if not no_csv:
        csv_dir = csv_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(csv_dir, exist_ok=True)
        long_path, wide_path = write_csv(rows, csv_dir)
        csv_paths = {"long": long_path, "wide": wide_path}
        print(f"[百度指数] CSV -> {long_path}")
        print(f"[百度指数] CSV -> {wide_path}")

    # 统计
    series = {}
    for kw, src, d, v in rows:
        key = f"{kw}_{src}"
        if key not in series:
            series[key] = {"n": 0, "sum": 0.0, "first": d, "last": d}
        series[key]["last"] = d
        if series[key]["n"] == 0:
            series[key]["first"] = d
        series[key]["n"] += 1
        if v is not None:
            series[key]["sum"] += v

    print("\n[百度指数] 入库概览")
    for key in sorted(series):
        s = series[key]
        avg = s["sum"] / s["n"] if s["n"] else 0
        print(f"  {key}: n={s['n']:5d}  mean={avg:12.2f}  {s['first']} ~ {s['last']}")

    return {
        "rows_count": n,
        "db_path": db_path,
        "csv_paths": csv_paths,
        "series": series,
    }


def main():
    parser = argparse.ArgumentParser(description="百度指数抓取入库（search_all + feed，日频）")
    parser.add_argument("--keywords", default=",".join(DEFAULT_KEYWORDS), help="逗号分隔的关键词，1 个或多个")
    parser.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD（不传则抓当年 1 月 1 日到今天）")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--cookie-file", default=None, help="从文件读取 Cookie（替代交互输入）")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB 文件路径")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    parser.add_argument("--no-csv", action="store_true", help="不写 CSV，只入库")
    args = parser.parse_args()

    run(
        keywords=args.keywords,
        start=args.start,
        end=args.end,
        cookie_file=args.cookie_file,
        db_path=args.db,
        headless=args.headless,
        no_csv=args.no_csv,
    )


if __name__ == "__main__":
    main()
