# -*- coding: utf-8 -*-
"""百度指数数据抓取入库脚本（搜索指数 + 资讯指数）。

用法：
    python baidu_index_ingest.py                       # 交互式粘贴 Cookie，抓全历史（周频）
    python baidu_index_ingest.py --cookie-file c.txt   # 从文件读 Cookie
    python baidu_index_ingest.py --keywords 金龙鱼,浪潮信息
    python baidu_index_ingest.py --start 2026-08-18    # 抓 2026-08-18 至今（日频）
    python baidu_index_ingest.py --start 2026-08-18 --end 2026-09-16
    python baidu_index_ingest.py --headless            # 无头模式（默认有头，更稳）

关键词：--keywords 逗号分隔，1 个或多个都行。
时间范围：不传 --start/--end 抓「全部」（周频全历史）；传了则抓指定窗口，
    接口按窗口长短自动返回日频或周频，脚本按响应里的 type 字段自动对齐日期。
重复入库：表主键为 (keyword, source, freq, date)，INSERT OR REPLACE 幂等覆盖，
    重复抓同一窗口不会产生重复行；日频与周频因 freq 不同而互不干扰。

依赖：
    pip install duckdb playwright
    playwright install chromium

为什么不能只用 Cookie 直接调接口：
    百度指数的 /api/SearchApi/index 和 /api/FeedSearchApi/getFeedIndex 要求请求头
    Cipher-Text，该值由页面里的百度「Paris」反爬 SDK（window.Paris.getAcsInstance().getSign()）
    在浏览器内基于设备指纹动态生成（约 5.5 小时有效），纯 Python 无法复现。
    因此脚本用 Playwright 启动浏览器、注入你提供的 Cookie 完成登录，再由页面内 JS 生成
    token 并直接 fetch 接口，抓回数据后解码、入库。

Cookie 怎么来：你自己在浏览器里登录 index.baidu.com 后，用 DevTools 抓包，
复制请求头里的 Cookie（含 BDUSS 等）整串给脚本即可，不是账号密码。
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

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baidu_index.duckdb")
DEFAULT_KEYWORDS = ["金龙鱼", "浪潮信息"]
TABLE_NAME = "baidu_index"

# 在页面内执行：取 Cipher-Text token → 抓搜索/资讯指数 → ptbk 解码 → 返回结构化数据
# arg = {keywords: [...], startDate: "YYYY-MM-DD"|null, endDate: "YYYY-MM-DD"|null}
SCRAPE_JS = r"""
async (arg) => {
  const keywords = arg.keywords;
  const startDate = arg.startDate || null;
  const endDate = arg.endDate || null;

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
  const headers = { credentials: 'include', headers: { 'Cipher-Text': token, 'Accept': 'application/json, text/plain, */*' } };
  const word = JSON.stringify(keywords.map(k => [{ name: k, wordType: 1 }]));
  const qs = new URLSearchParams({ area: '0', word });
  if (startDate) qs.set('startDate', startDate);
  if (endDate) qs.set('endDate', endDate);
  const searchUrl = 'https://index.baidu.com/api/SearchApi/index?' + qs.toString();
  const feedUrl = 'https://index.baidu.com/api/FeedSearchApi/getFeedIndex?' + qs.toString();

  const search = await (await fetch(searchUrl, headers)).json();
  const feed = await (await fetch(feedUrl, headers)).json();
  if (search.status !== 0) throw new Error('SearchApi status=' + search.status + ' ' + (search.message || ''));
  if (feed.status !== 0) throw new Error('FeedSearchApi status=' + feed.status + ' ' + (feed.message || ''));

  async function getPtbk(uniqid) {
    const r = await (await fetch('https://index.baidu.com/Interface/ptbk?uniqid=' + uniqid, headers)).json();
    if (r.status !== 0 || !r.data) throw new Error('ptbk status=' + r.status);
    return r.data;
  }

  const searchPtbk = await getPtbk(search.data.uniqid);
  const feedPtbk = feed.data.uniqid === search.data.uniqid ? searchPtbk : await getPtbk(feed.data.uniqid);

  const payload = { search: [], feed: [] };
  (search.data.userIndexes || []).forEach(ui => {
    const kw = (ui.word || []).map(x => x.name).join('');
    const freq = ui.type || 'day';
    for (const k of ['all', 'pc', 'wise']) {
      const blk = ui[k];
      payload.search.push({
        keyword: kw, series: k, freq: freq,
        startDate: blk.startDate, endDate: blk.endDate,
        values: decrypt(searchPtbk, blk.data).split(','),
      });
    }
  });
  (feed.data.index || []).forEach(it => {
    const kw = (it.key || []).map(x => x.name).join('');
    payload.feed.push({
      keyword: kw, freq: it.type || 'day',
      startDate: it.startDate, endDate: it.endDate,
      values: decrypt(feedPtbk, it.data).split(','),
    });
  });
  return payload;
}
"""


def _require_duckdb():
    if duckdb is None:
        raise ImportError("缺少 duckdb，请安装：python3 -m pip install duckdb")


def _require_playwright():
    if sync_playwright is None:
        raise ImportError("缺少 playwright，请安装：python3 -m pip install playwright && playwright install chromium")


def parse_cookie_text(text):
    """把用户粘贴的 Cookie 解析成 {name: value}，兼容三种格式。"""
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("["):  # JSON 数组（浏览器扩展导出）
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


def scrape(keywords, cookie_text, start_date=None, end_date=None, headless=False):
    """启动浏览器注入 Cookie，抓取并解码数据，返回 payload dict。"""
    _require_playwright()
    pairs = parse_cookie_text(cookie_text)
    if not pairs:
        raise ValueError("未解析到任何 Cookie，请检查粘贴内容（应包含 BDUSS 等）。")
    cookies = [{"name": k, "value": v, "domain": ".baidu.com", "path": "/"} for k, v in pairs.items()]

    arg = {"keywords": keywords, "startDate": start_date, "endDate": end_date}
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


def dates_for(start, end, freq):
    """按频率生成日期序列：day=逐日，week=周一对齐逐周。"""
    s = datetime.date.fromisoformat(start)
    e = datetime.date.fromisoformat(end)
    out = []
    if freq == "day":
        d = s
        while d <= e:
            out.append(d.isoformat())
            d += datetime.timedelta(days=1)
        return out
    s -= datetime.timedelta(days=s.weekday())
    e -= datetime.timedelta(days=e.weekday())
    d = s
    while d <= e:
        out.append(d.isoformat())
        d += datetime.timedelta(days=7)
    return out


def payload_to_rows(payload):
    """把 payload 展平成 (keyword, source, freq, date, value) 行，空值记为 None。"""
    rows = []
    for item in payload.get("search", []):
        src = "search_" + item["series"]
        dates = dates_for(item["startDate"], item["endDate"], item["freq"])
        for d, v in zip(dates, item["values"]):
            rows.append((item["keyword"], src, item["freq"], d, None if v == "" else float(v)))
    for item in payload.get("feed", []):
        dates = dates_for(item["startDate"], item["endDate"], item["freq"])
        for d, v in zip(dates, item["values"]):
            rows.append((item["keyword"], "feed", item["freq"], d, None if v == "" else float(v)))
    return rows


def ingest(rows, db_path=DEFAULT_DB_PATH):
    """UPSERT 到 DuckDB 的 baidu_index 表。"""
    _require_duckdb()
    con = duckdb.connect(db_path)
    try:
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                keyword TEXT,
                source TEXT,
                freq TEXT,
                date DATE,
                value DOUBLE,
                PRIMARY KEY (keyword, source, freq, date)
            )
            """
        )
        con.executemany(
            f"INSERT OR REPLACE INTO {TABLE_NAME} (keyword, source, freq, date, value) VALUES (?, ?, ?, ?, ?)",
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
        w.writerow(["keyword", "source", "freq", "date", "value"])
        w.writerows(rows)

    wide = {}
    dates = set()
    for kw, src, freq, d, v in rows:
        wide.setdefault(f"{kw}_{src}_{freq}", {})[d] = v
        dates.add(d)
    colnames = sorted(wide)
    dates_sorted = sorted(dates)
    with open(wide_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date"] + colnames)
        for d in dates_sorted:
            w.writerow([d] + [wide[c].get(d, "") for c in colnames])
    return long_path, wide_path


def main():
    parser = argparse.ArgumentParser(description="百度指数抓取入库（搜索指数+资讯指数）")
    parser.add_argument("--keywords", default=",".join(DEFAULT_KEYWORDS), help="逗号分隔的关键词，1 个或多个")
    parser.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD（不传则抓全历史周频）")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--cookie-file", default=None, help="从文件读取 Cookie（替代交互输入）")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB 文件路径")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    parser.add_argument("--no-csv", action="store_true", help="不写 CSV，只入库")
    args = parser.parse_args()

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    if not keywords:
        print("未提供关键词。")
        sys.exit(1)

    today = datetime.date.today().isoformat()
    start_date = args.start
    end_date = args.end or today
    if start_date is None and args.end is None:
        start_date, end_date = None, None  # 全部（周频）
    elif start_date is None:
        start_date = (datetime.date.fromisoformat(end_date) - datetime.timedelta(days=30)).isoformat()

    if args.cookie_file:
        with open(args.cookie_file, encoding="utf-8") as f:
            cookie_text = f.read()
    else:
        cookie_text = read_cookie_interactive()

    window = "全部（周频）" if start_date is None else f"{start_date} ~ {end_date}"
    print(f"关键词：{keywords}，时间范围：{window}")
    payload = scrape(keywords, cookie_text, start_date=start_date, end_date=end_date, headless=args.headless)

    rows = payload_to_rows(payload)
    n = ingest(rows, db_path=args.db)
    print(f"入库 {n} 行 -> {args.db}")

    if not args.no_csv:
        long_path, wide_path = write_csv(rows, os.path.dirname(os.path.abspath(__file__)))
        print(f"CSV -> {long_path}")
        print(f"CSV -> {wide_path}")

    print("\n=== 入库概览 ===")
    series = {}
    for kw, src, freq, d, v in rows:
        key = f"{kw}_{src}_{freq}"
        if key not in series:
            series[key] = {"n": 0, "sum": 0.0, "first": d, "last": d}
        series[key]["last"] = d
        if series[key]["n"] == 0:
            series[key]["first"] = d
        series[key]["n"] += 1
        if v is not None:
            series[key]["sum"] += v
    for key in sorted(series):
        s = series[key]
        avg = s["sum"] / s["n"] if s["n"] else 0
        print(f"{key}: n={s['n']:5d}  mean={avg:12.2f}  {s['first']} ~ {s['last']}")


if __name__ == "__main__":
    main()
