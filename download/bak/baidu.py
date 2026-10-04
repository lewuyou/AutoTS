# -*- coding: utf-8 -*-
"""百度指数数据抓取入库模块（搜索指数 search_all + 资讯指数 feed，日频）。

可独立运行，也可由 download_all.py 调用 run()。

用法（独立运行）：
    python -m download.baidu                          # 交互式粘贴 Cookie，增量抓默认关键词
    python -m download.baidu --cookie-file c.txt      # 从文件读 Cookie
    python -m download.baidu --keywords 金龙鱼,浪潮信息
    python -m download.baidu --from-stock-list        # 从 stock_list 表读全部股票简称循环抓取
    python -m download.baidu --start 2013-01-01       # 强制指定窗口（忽略库内增量起点）
    python -m download.baidu --headless               # 无头模式
    python -m download.baidu --blacklist-file b.txt   # 指定黑名单文件（默认 baidu_blacklist.txt）
    python -m download.baidu --no-exclude-prefix      # 不排除 C/N/XD/XR/DR/S 临时前缀名

依赖：
    pip install duckdb playwright
    playwright install chromium

全量/增量规则（不传 --start 时）：
    每个关键词独立计算起点 = max(指数收录起始日, 库内该关键词最大日期+1)，
    库内无记录即全量（search_all 自 2011-01-01，feed 自 2017-07-03）；
    关键词按相同 (search起点, feed起点) 分组，每组最多 batch_size 个词一批请求
    （百度指数对比接口本身支持多词），每批抓完立即入库，中断后重跑自动续抓。
    结束日默认按数据滞后取：search 截至昨日、feed 截至前日，避免空查尚未发布的新一天。
    单批内某词未收录导致整批报错时，页面内自动逐词隔离重试，不影响同组其他词。
    未收录关键词（百度指数无该词数据）自动加入黑名单，下次运行不再查询；
    失败关键词（风控/token 过期/抓取异常）写入 download/temp/baidu_failed_<时间戳>.txt，下轮重试。

夜间暂停：
    每日 00:00 ~ 08:00 自动暂停抓取（百度指数夜间不更新新一天数据，且更易触发风控），
    到 08:00 自动恢复继续爬取。

暂存模式（--stage）：
    每批数据不入主库，改写入暂存 DuckDB（表结构与主库 baidu 表完全相同），
    供多个并行任务各自生成暂存文件后，用 --merge 统一合并入主库：
        python -m download.baidu --merge "download/temp/baidu_stage_*.duckdb"
    合并成功的暂存文件自动重命名加 .merged 后缀，防止重复合并。
    注意：暂存模式下增量起点取 主库与暂存库 的较大日期（两边都读），中断重跑可续抓。

入库表 baidu 字段含义：
    keyword  搜索关键词（股票简称，如 "金龙鱼"）
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
DEFAULT_COOKIE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baidu_cookie.txt")
DEFAULT_BLACKLIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baidu_blacklist.txt")
DEFAULT_KEYWORDS = ["金龙鱼"]
TABLE_NAME = "baidu"
# 临时状态名前缀：C=注册制次新股、N=新股上市首日、XD=除息、XR=除权、DR=除权除息、S=未股改。
# 这些名称只在特定交易日出现，次日即恢复正常简称，百度指数基本无数据，按前缀直接跳过。
TEMP_NAME_PREFIXES = ("XD", "XR", "DR", "C", "N", "S")
SEARCH_START = "2011-01-01"   # search_all（整体=PC+移动）最早日期
FEED_START = "2017-07-03"     # 资讯指数最早日期
SEARCH_START_D = datetime.date.fromisoformat(SEARCH_START)
FEED_START_D = datetime.date.fromisoformat(FEED_START)
CHUNK_DAYS = 360              # 日频单次请求最大天数（实测 365 内返回日频），留余量
BATCH_SIZE = 1                # 单次请求最多对比关键词数
REQUEST_SLEEP_MS = 5000       # 每个请求段间隔毫秒基数（实际为 5000~8000 随机，反爬安全间隔）
BLOCK_COOLDOWN_S = 3600       # 触发风控(10001)后冷却秒数（1 小时）

# 页面内执行：取 token → 分段抓 search_all / feed → ptbk 解码 → 返回 {search, feed, errors}
# arg = {keywords: [...], searchChunks: [[start,end],...], feedChunks: [[start,end],...], sleepMs: 5000}
# 单段失败不中断：记入 errors 继续；整批报错且多词时逐词隔离重试；token 过期(10018)自动重取。
SCRAPE_JS = r"""
async (arg) => {
  const keywords = arg.keywords;
  const searchChunks = arg.searchChunks || [];
  const feedChunks = arg.feedChunks || [];
  const SLEEP = arg.sleepMs || 3000;
  const ISOLATE_SLEEP = Math.max(300, Math.floor(SLEEP / 2));
  const errors = [];
  // 每次间隔在 SLEEP ~ SLEEP+3000 之间随机，模拟人工操作防风控
  const randSleep = () => SLEEP + Math.floor(Math.random() * 3000);

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

  let token = await getToken();
  const H = () => ({ credentials: 'include', headers: { 'Cipher-Text': token, 'Accept': 'application/json, text/plain, */*' } });
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  let blockCount = 0;
  // 凌晨 00:00 ~ 08:00 暂停爬取：百度指数夜间不更新新一天数据，且夜间请求更易触发风控
  const NIGHT_END_HOUR = 8;
  const pauseIfNight = async () => {
    const now = new Date();
    if (now.getHours() < NIGHT_END_HOUR) {
      const resume = new Date(now);
      resume.setHours(NIGHT_END_HOUR, 0, 0, 0);
      const waitMs = resume - now;
      console.log(`[百度指数] 夜间暂停（${now.toTimeString().slice(0, 8)}），等待至 ${resume.toTimeString().slice(0, 8)}，约 ${Math.round(waitMs / 60000)} 分钟`);
      await sleep(Math.max(waitMs, 1000));
    }
  };

  async function fetchJson(url) {
    let r = await (await fetch(url, H())).json();
    if (r && r.status === 10018) {
      token = await getToken();
      r = await (await fetch(url, H())).json();
    }
    if (r && r.status === 10001) {
      // 风控限流：第一次触发冷却 1 小时后重试；第二次触发直接中止脚本
      blockCount++;
      if (blockCount >= 2) throw new Error('触发百度风控(10001)第 2 次，已中止脚本，请冷却后重跑（自动断点续抓）');
      console.log('[百度指数] 触发风控(10001)，冷却 1 小时...');
      await sleep(3600000);
      r = await (await fetch(url, H())).json();
      if (r && r.status === 10001) {
        blockCount++;
        throw new Error('触发百度风控(10001)第 2 次（冷却 1 小时后重试仍被限），已中止脚本，请冷却后重跑（自动断点续抓）');
      }
    }
    return r;
  }

  async function getPtbk(uniqid) {
    const r = await fetchJson('https://index.baidu.com/Interface/ptbk?uniqid=' + uniqid);
    if (r.status !== 0 || !r.data) throw new Error('ptbk status=' + r.status);
    return r.data;
  }

  const wordParam = kws => JSON.stringify(kws.map(k => [{ name: k, wordType: 1 }]));

  async function runChunks(kind, baseUrl, chunks, extract) {
    const out = {};
    const badWords = new Set();
    for (const [s, e] of chunks) {
      await pauseIfNight();
      const tag = s + '~' + e;
      const qs = new URLSearchParams({ area: '0', word: wordParam(keywords), startDate: s, endDate: e });
      let r;
      try {
        r = await fetchJson(baseUrl + qs.toString());
      } catch (err) {
        if (String(err).includes('已中止')) throw err;
        errors.push({ kind, keywords, chunk: tag, message: String(err) });
        await sleep(randSleep());
        continue;
      }
      if (r.status === 0) {
        try { await extract(r, out); } catch (err) {
          if (String(err).includes('已中止')) throw err;
          errors.push({ kind, keywords, chunk: tag, message: String(err) });
        }
      } else if (keywords.length > 1) {
        for (const kw of keywords) {
          if (badWords.has(kw)) continue;
          const qs1 = new URLSearchParams({ area: '0', word: wordParam([kw]), startDate: s, endDate: e });
          try {
            const r1 = await fetchJson(baseUrl + qs1.toString());
            if (r1.status !== 0) {
              errors.push({ kind, keywords: [kw], chunk: tag, status: r1.status, message: r1.message || '' });
              // 未收录类错误后续时间段直接跳过该词，避免逐段重复请求触发风控
              if (r1.status !== 10001) badWords.add(kw);
            } else {
              try { await extract(r1, out); } catch (err) {
                if (String(err).includes('已中止')) throw err;
                errors.push({ kind, keywords: [kw], chunk: tag, message: String(err) });
              }
            }
          } catch (err) {
            if (String(err).includes('已中止')) throw err;
            errors.push({ kind, keywords: [kw], chunk: tag, message: String(err) });
          }
          await sleep(ISOLATE_SLEEP);
        }
      } else {
        errors.push({ kind, keywords, chunk: tag, status: r.status, message: r.message || '' });
      }
      await sleep(randSleep());
    }
    return out;
  }

  const search = await runChunks('search', 'https://index.baidu.com/api/SearchApi/index?', searchChunks, async (r, out) => {
    const ptbk = await getPtbk(r.data.uniqid);
    for (const ui of (r.data.userIndexes || [])) {
      const kw = (ui.word || []).map(x => x.name).join('');
      (out[kw] = out[kw] || []).push({
        start: ui.all.startDate, end: ui.all.endDate,
        values: decrypt(ptbk, ui.all.data).split(','),
      });
    }
  });

  const feed = await runChunks('feed', 'https://index.baidu.com/api/FeedSearchApi/getFeedIndex?', feedChunks, async (r, out) => {
    const ptbk = await getPtbk(r.data.uniqid);
    for (const it of (r.data.index || [])) {
      const kw = (it.key || []).map(x => x.name).join('');
      (out[kw] = out[kw] || []).push({
        start: it.startDate, end: it.endDate,
        values: decrypt(ptbk, it.data).split(','),
      });
    }
  });

  return { search, feed, errors };
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


def load_stock_names(db_path):
    """从 stock_list 表读取全部股票简称（去重）。"""
    _require_duckdb()
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            "SELECT DISTINCT name FROM stock_list WHERE name IS NOT NULL AND name <> '' ORDER BY name"
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def _normalize_name(name):
    """去掉所有空白字符（含全角空格），用于黑名单匹配时消除 '七 匹 狼'/'七匹狼' 之类差异。"""
    return "".join(str(name).split())


def load_blacklist(path=DEFAULT_BLACKLIST_FILE):
    """从黑名单文件读取股票简称集合（一行一个，# 开头为注释，自动去空白去重）。文件不存在返回空集。"""
    names = set()
    if not path or not os.path.exists(path):
        return names
    with open(path, encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            names.add(_normalize_name(name))
    return names


def add_to_blacklist(names, path=DEFAULT_BLACKLIST_FILE):
    """把未收录关键词追加到黑名单文件（一行一个，去重，已在文件中的跳过）。

    返回本次新增写入的词数；path 为空或 names 为空时返回 0（不写）。
    若文件末尾无换行，先补一个换行，避免新名字拼接到最后一行。
    """
    if not path or not names:
        return 0
    existing = load_blacklist(path)
    new = []
    for n in names:
        if _normalize_name(n) not in existing:
            new.append(n)
            existing.add(_normalize_name(n))
    if not new:
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    prefix = ""
    if os.path.exists(path):
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    prefix = "\n"
    with open(path, "a", encoding="utf-8") as f:
        if prefix:
            f.write(prefix)
        for n in new:
            f.write(n + "\n")
    return len(new)


def load_last_dates(db_path):
    """{keyword: {source: 库内最大日期}}；baidu 表不存在时返回空（即全量）。"""
    _require_duckdb()
    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
        if TABLE_NAME not in tables:
            return {}
        rows = con.execute(
            f"SELECT keyword, source, MAX(date) FROM {TABLE_NAME} GROUP BY keyword, source"
        ).fetchall()
    finally:
        con.close()
    out = {}
    for kw, src, d in rows:
        out.setdefault(kw, {})[src] = d
    return out


def plan_batches(keywords, search_end_d, feed_end_d, last_dates, batch_size=BATCH_SIZE, sort_by_lag=True):
    """每股起点 = max(收录起始日, 库内最大日期+1)，按相同起点分组后按 batch_size 切批。

    search/feed 各用独立的结束日（百度指数 search 数据滞后 1 天、feed 滞后 2 天），
    增量时不查尚未发布的日期，避免每次空查不存在的新一天。
    sort_by_lag=True 时按与主库数据差距时间排序：差距越大（起点越早）越靠前，
    优先补抓滞后最久的关键词；False 则保持传入 keywords 的原始顺序。
    返回 (batches, skipped)；batches 元素为 (keywords, search_chunks, feed_chunks)，
    skipped 为库内已覆盖到各自结束日的关键词。
    """
    # 百度 API 会把英文子串转小写（如 TCL中环 -> tcl中环），按小写归一化匹配，
    # 避免库内已抓词因大小写不一致被当成新词全量重抓。
    folded = {}
    for k, srcs in last_dates.items():
        key = k.lower()
        for src, d in srcs.items():
            cur = folded.setdefault(key, {}).get(src)
            if cur is None or d > cur:
                folded[key][src] = d

    groups = {}
    skipped = []
    first_seen = {}
    for idx, kw in enumerate(keywords):
        info = folded.get(kw.lower(), {})
        sd = info.get("search_all")
        ss = max(SEARCH_START_D, sd + datetime.timedelta(days=1)) if sd else SEARCH_START_D
        fd = info.get("feed")
        fs = max(FEED_START_D, fd + datetime.timedelta(days=1)) if fd else FEED_START_D
        if ss > search_end_d and fs > feed_end_d:
            skipped.append(kw)
            continue
        key = (ss, fs)
        groups.setdefault(key, []).append(kw)
        first_seen.setdefault(key, idx)
    if sort_by_lag:
        # 结束日对全词一致，故起点(ss/fs)越早 = 与主库差距越大，升序即差距降序 → 滞后最久优先
        ordered_keys = sorted(groups, key=lambda k: (k[0], k[1]))
    else:
        ordered_keys = sorted(groups, key=first_seen.get)
    batches = []
    for key in ordered_keys:
        ss, fs = key
        kws = groups[key]
        sc = build_chunks(ss.isoformat(), search_end_d.isoformat()) if ss <= search_end_d else []
        fc = build_chunks(fs.isoformat(), feed_end_d.isoformat()) if fs <= feed_end_d else []
        for i in range(0, len(kws), batch_size):
            batches.append((kws[i:i + batch_size], sc, fc))
    return batches, skipped


def scrape_batches(batch_args, cookie_text, headless=False, on_batch=None):
    """启动一次浏览器，顺序执行所有批次；每批完成回调 on_batch(idx, arg, payload)。"""
    _require_playwright()
    pairs = parse_cookie_text(cookie_text)
    if not pairs:
        raise ValueError("未解析到任何 Cookie，请检查粘贴内容（应包含 BDUSS 等）。")
    cookies = [{"name": k, "value": v, "domain": ".baidu.com", "path": "/"} for k, v in pairs.items()]

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
        for i, arg in enumerate(batch_args):
            payload = page.evaluate(SCRAPE_JS, arg)
            if on_batch:
                on_batch(i, arg, payload)
        browser.close()


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
    """UPSERT 到 DuckDB 的 baidu 表（表不存在先建表）。空列表只建表不写行。"""
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
        if rows:
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


def merge_stages(pattern, db_path=DEFAULT_DB_PATH):
    """把暂存 DuckDB 文件统一合并入主库 baidu 表；已合并文件加 .merged 后缀。

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
                keyword TEXT,
                source TEXT,
                date DATE,
                value DOUBLE,
                PRIMARY KEY (keyword, source, date)
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
                print(f"[百度指数] 合并 {f}: 0 行（空暂存）")
                continue
            n = con.execute(f"SELECT COUNT(*) FROM stage.{TABLE_NAME}").fetchone()[0]
            con.execute(
                f"INSERT OR REPLACE INTO {TABLE_NAME} (keyword, source, date, value) "
                f"SELECT keyword, source, date, value FROM stage.{TABLE_NAME}"
            )
            con.execute("DETACH stage")
            os.rename(f, f + ".merged")
            total += n
            print(f"[百度指数] 合并 {f}: {n} 行")
    finally:
        con.close()
    print(f"[百度指数] 合并完成：{len(files)} 个文件，共 {total} 行 -> {db_path}")
    return total, len(files)


def run(keywords=None, start=None, end=None, cookie_file=DEFAULT_COOKIE_FILE, db_path=None, headless=False,
        no_csv=False, csv_dir=None, from_stock_list=False, batch_size=BATCH_SIZE,
        sleep_ms=REQUEST_SLEEP_MS, stage_path=None, exclude_st=True,
        blacklist_file=DEFAULT_BLACKLIST_FILE, exclude_prefix=True, sort_by_lag=True):
    """供外部调用的入口。

    参数：
        keywords: 关键词列表，默认 ["金龙鱼"]
        start: 起始日期 YYYY-MM-DD；不传则按库内最大日期增量（无记录即 2011 起全量）
        end: 结束日期 YYYY-MM-DD；不传则按数据滞后规则（search 截至昨日、feed 截至前日）
        cookie_file: Cookie 文件路径，None 则交互式输入
        db_path: DuckDB 文件路径，默认模块目录下 autots.duckdb
        headless: 是否无头模式
        no_csv: 是否跳过 CSV 导出（from_stock_list 时强制跳过，数据量太大）
        csv_dir: CSV 输出目录，默认模块目录下 temp/
        from_stock_list: 从 stock_list 表读取全部股票简称作为关键词
        batch_size: 单次请求对比关键词数上限，默认 3
        sleep_ms: 每个请求段间隔毫秒基数，默认 5000（实际 5000~8000 随机，反爬安全间隔）
        stage_path: 暂存 DuckDB 路径；"auto" 自动生成 download/temp/baidu_stage_<时间戳>_<pid>.duckdb；
                    设置后每批写入暂存库而非主库，事后用 merge_stages 统一合并
        exclude_st: 排除名称含 ST 的股票（*ST/ST 股百度指数噪声大且大量未收录）
        blacklist_file: 黑名单文件路径（一行一个股票简称，匹配时忽略空白），None 或文件不存在则跳过；
                        抓取中识别为未收录的关键词会自动追加进该文件，下次运行自动跳过
        exclude_prefix: 排除 C/N/XD/XR/DR/S 开头的临时状态名（默认 True）
        sort_by_lag: 按与主库数据差距时间排序抓取顺序，差距越大越靠前（默认 True）；
                     传 start 显式指定窗口时不适用（全词同窗口）
    返回：
        dict 包含 rows_count, db_path, failed_path 等
    """
    db_path = db_path or DEFAULT_DB_PATH
    if from_stock_list:
        keywords = load_stock_names(db_path)
        print(f"[百度指数] 从 stock_list 读取 {len(keywords)} 个股票简称 -> {db_path}")
    if keywords is None:
        keywords = DEFAULT_KEYWORDS
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    keywords = list(dict.fromkeys(keywords))
    if exclude_st:
        before = len(keywords)
        keywords = [k for k in keywords if "ST" not in k.upper()]
        print(f"[百度指数] 排除 ST 股 {before - len(keywords)} 个，剩余 {len(keywords)}")
    blacklist = load_blacklist(blacklist_file)
    if blacklist:
        before = len(keywords)
        keywords = [k for k in keywords if _normalize_name(k) not in blacklist]
        print(f"[百度指数] 排除黑名单 {before - len(keywords)} 个，剩余 {len(keywords)}")
    if exclude_prefix:
        before = len(keywords)
        keywords = [k for k in keywords if not k.upper().startswith(TEMP_NAME_PREFIXES)]
        print(f"[百度指数] 排除临时前缀名(C/N/XD/XR/DR/S) {before - len(keywords)} 个，剩余 {len(keywords)}")
    if not keywords:
        raise ValueError("未提供关键词")

    if stage_path == "auto":
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stage_path = os.path.join(temp_dir, f"baidu_stage_{ts}_{os.getpid()}.duckdb")
    if stage_path:
        print(f"[百度指数] 暂存模式：每批写入 {stage_path}，事后用 --merge 合并入主库")
    ingest_db = stage_path or db_path

    today = datetime.date.today()
    if end:
        search_end_d = feed_end_d = datetime.date.fromisoformat(end)
    else:
        # 百度指数数据滞后：search 截至昨日、feed 截至前日，增量时不空查尚未发布的新一天
        search_end_d = today - datetime.timedelta(days=1)
        feed_end_d = today - datetime.timedelta(days=2)

    if start:
        start_d = datetime.date.fromisoformat(start)
        sc = build_chunks(start_d.isoformat(), search_end_d.isoformat()) if start_d <= search_end_d else []
        feed_start = max(start_d, FEED_START_D)
        fc = build_chunks(feed_start.isoformat(), feed_end_d.isoformat()) if feed_start <= feed_end_d else []
        batches = [(keywords[i:i + batch_size], sc, fc) for i in range(0, len(keywords), batch_size)]
        skipped = []
    else:
        last = load_last_dates(db_path)
        if stage_path and os.path.exists(stage_path):
            for kw, srcs in load_last_dates(stage_path).items():
                for src, d in srcs.items():
                    cur = last.setdefault(kw, {}).get(src)
                    if cur is None or d > cur:
                        last[kw][src] = d
        batches, skipped = plan_batches(keywords, search_end_d, feed_end_d, last, batch_size, sort_by_lag=sort_by_lag)

    if not batches:
        print(f"[百度指数] {len(skipped)} 个关键词库内均已覆盖到结束日（search={search_end_d}, feed={feed_end_d}），无需抓取")
        return {"rows_count": 0, "db_path": db_path, "csv_paths": None, "series": {}, "failed_path": None}

    if cookie_file:
        with open(cookie_file, encoding="utf-8") as f:
            cookie_text = f.read()
    else:
        cookie_text = read_cookie_interactive()

    total = len(batches)
    print(f"[百度指数] 关键词 {len(keywords)} 个，共 {total} 批（每批≤{batch_size} 词），结束日 search={search_end_d} / feed={feed_end_d}")
    if skipped:
        print(f"[百度指数] {len(skipped)} 个已是最新，跳过")

    stats = {}
    failed = {}
    unindexed = set()   # 本次已识别并写入黑名单的未收录词（去重）
    total_rows = [0]

    def on_batch(i, arg, payload):
        rows = payload_to_rows(payload)
        n = ingest(rows, db_path=ingest_db)
        total_rows[0] += n
        returned = set(payload.get("search", {})) | set(payload.get("feed", {}))
        errors_by_kw = {}
        for err in payload.get("errors", []):
            msg = f"{err.get('kind')} {err.get('chunk', '')} status={err.get('status', '')} {err.get('message', '')}".strip()
            for kw in err.get("keywords", arg["keywords"]):
                errors_by_kw.setdefault(kw, []).append((err.get("status"), msg))
        batch_unindexed = []
        for kw in arg["keywords"]:
            if kw in returned or kw in unindexed:
                continue
            kw_errs = errors_by_kw.get(kw, [])
            # 10001=风控、10018=token 过期、status=None=抓取异常：临时失败，下轮重试，不判未收录
            if any(status is None or status in (10001, 10018) for status, _ in kw_errs):
                for _, msg in kw_errs:
                    failed.setdefault(kw, set()).add(msg)
            else:
                batch_unindexed.append(kw)
        for kw, src, d, v in rows:
            key = f"{kw}_{src}"
            s = stats.setdefault(key, {"n": 0, "sum": 0.0, "first": d, "last": d})
            s["last"] = d
            s["n"] += 1
            if v is not None:
                s["sum"] += v
        if batch_unindexed:
            added = add_to_blacklist(batch_unindexed, blacklist_file)
            unindexed.update(batch_unindexed)
            print(f"[百度指数] 未收录 {len(batch_unindexed)} 词，加入黑名单（本次新增 {added}）: {'、'.join(batch_unindexed)}", flush=True)
        print(f"[百度指数] 批 {i + 1}/{total} [{'、'.join(arg['keywords'])}] 入库 {n} 行，累计 {total_rows[0]}", flush=True)

    batch_args = [{"keywords": kws, "searchChunks": sc, "feedChunks": fc, "sleepMs": sleep_ms}
                  for kws, sc, fc in batches]
    scrape_batches(batch_args, cookie_text, headless=headless, on_batch=on_batch)

    if unindexed:
        print(f"[百度指数] 本次共识别 {len(unindexed)} 个未收录关键词，已写入黑名单 {blacklist_file}，下次不再查询")

    failed_path = None
    if failed:
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_path = os.path.join(temp_dir, f"baidu_failed_{ts}.txt")
        with open(failed_path, "w", encoding="utf-8") as f:
            for kw in sorted(failed):
                f.write(f"{kw}\t{'; '.join(sorted(failed[kw]))}\n")
        print(f"[百度指数] {len(failed)} 个关键词失败（风控/token/异常，下轮重试） -> {failed_path}")

    csv_paths = None
    if not no_csv and not from_stock_list:
        all_rows = []
        con = duckdb.connect(ingest_db, read_only=True)
        try:
            all_rows = con.execute(
                f"SELECT keyword, source, date, value FROM {TABLE_NAME} WHERE keyword IN ({','.join(['?'] * len(keywords))})",
                keywords,
            ).fetchall()
        finally:
            con.close()
        csv_dir = csv_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(csv_dir, exist_ok=True)
        long_path, wide_path = write_csv(all_rows, csv_dir)
        csv_paths = {"long": long_path, "wide": wide_path}
        print(f"[百度指数] CSV -> {long_path}")

    print(f"\n[百度指数] 完成：共入库 {total_rows[0]} 行，覆盖 {len(stats)} 个序列")
    if len(stats) <= 50:
        for key in sorted(stats):
            s = stats[key]
            avg = s["sum"] / s["n"] if s["n"] else 0
            print(f"  {key}: n={s['n']:5d}  mean={avg:12.2f}  {s['first']} ~ {s['last']}")

    return {
        "rows_count": total_rows[0],
        "db_path": db_path,
        "stage_path": stage_path,
        "csv_paths": csv_paths,
        "series": stats,
        "failed_path": failed_path,
        "unindexed": len(unindexed),
    }


def main():
    parser = argparse.ArgumentParser(description="百度指数抓取入库（search_all + feed，日频）")
    parser.add_argument("--keywords", default=None, help="逗号分隔的关键词，1 个或多个")
    parser.add_argument("--from-stock-list", action="store_true", help="从 stock_list 表读取全部股票简称")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"单次请求对比关键词数，默认 {BATCH_SIZE}")
    parser.add_argument("--sleep-ms", type=int, default=REQUEST_SLEEP_MS, help=f"每个请求段间隔毫秒，默认 {REQUEST_SLEEP_MS}")
    parser.add_argument("--exclude-st", action="store_true", default=True, help="排除名称含 ST 的股票（默认开启）")
    parser.add_argument("--include-st", action="store_true", help="包含 ST 股（覆盖默认排除）")
    parser.add_argument("--blacklist-file", default=DEFAULT_BLACKLIST_FILE,
                        help=f"黑名单文件路径（一行一个股票简称），默认 {DEFAULT_BLACKLIST_FILE}；None/文件不存在则跳过")
    parser.add_argument("--no-exclude-prefix", action="store_true",
                        help="不排除 C/N/XD/XR/DR/S 开头的临时状态名（默认排除）")
    parser.add_argument("--no-sort-by-lag", action="store_true",
                        help="不按与主库数据差距时间排序（默认按差距越大越靠前）")
    parser.add_argument("--stage", default=None,
                        help="暂存 DuckDB 路径；auto 自动生成 download/temp/baidu_stage_<时间戳>_<pid>.duckdb；不设则直入主库")
    parser.add_argument("--merge", default=None,
                        help="合并模式：暂存文件 glob，如 \"download/temp/baidu_stage_*.duckdb\"，合并入 --db 后退出")
    parser.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD（不传则按库内最大日期增量，无记录全量）")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--cookie-file", default=DEFAULT_COOKIE_FILE, help=f"从文件读取 Cookie（默认 {DEFAULT_COOKIE_FILE}）")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB 文件路径")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    parser.add_argument("--no-csv", action="store_true", help="不写 CSV，只入库")
    args = parser.parse_args()

    if args.merge:
        merge_stages(args.merge, db_path=args.db)
        return

    run(
        keywords=args.keywords,
        start=args.start,
        end=args.end,
        cookie_file=args.cookie_file,
        db_path=args.db,
        headless=args.headless,
        no_csv=args.no_csv,
        from_stock_list=args.from_stock_list,
        batch_size=args.batch_size,
        sleep_ms=args.sleep_ms,
        stage_path=args.stage,
        exclude_st=not args.include_st,
        blacklist_file=args.blacklist_file,
        exclude_prefix=not args.no_exclude_prefix,
        sort_by_lag=not args.no_sort_by_lag,
    )


if __name__ == "__main__":
    main()
