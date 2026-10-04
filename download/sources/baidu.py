# -*- coding: utf-8 -*-
"""百度指数抓取（纯 API 版，无浏览器）：mini_racer 本地算 Cipher-Text token + requests 直连，日频。

取数通道（2026-10-03 起替代 Playwright 版，旧版备份在 download/bak/baidu.py）：
  - token：百度 Paris/ACS SDK（同目录 baidu_acs_2057.js，每次运行自动从百度 CDN 拉最新覆盖，
    下载失败回退本地缓存）只依赖 window/document.cookie/location/navigator 与时间，无环境指纹，故用 py_mini_racer（内嵌 V8）本地执行 $BSB_2057.gs() 直接出签名。
    签名不绑定 URL/账号，实测约 5.5 小时有效，缓存 4 小时或服务端返回 10018 时自动重算。
  - 请求：requests.Session 带 Cookie 头直连 index.baidu.com，风控策略：10001 冷却重试最多 3 次
    （1h -> 2h -> 4h 翻倍），第 4 次仍被限则中止；凌晨 00:00~08:00 暂停（指数夜间不更新且易风控）；
    每连续抓取 30 分钟休息 1 小时；每请求段 3~5 秒随机间隔。
数据源特有逻辑仅保留：token 生成 + 分段抓取 + ptbk 解码、黑名单/临时名前缀过滤、
未收录词自动入黑名单、关键词滞后排序与分批、search/feed 独立滞后窗口。
公共流程（日志/建表/UPSERT/水位/暂存路径/合并/结果）走 download.common。

依赖：
    pip install py_mini_racer requests

用法（独立运行）：
    python -m download.sources.baidu                          # 默认关键词
    python -m download.sources.baidu --from-stock-list        # 全部股票简称
    python -m download.sources.baidu --keywords 金龙鱼,浪潮信息
    python -m download.sources.baidu --stage auto             # 暂存模式
    python -m download.sources.baidu --merge "download/temp/baidu_stage_*.duckdb"

入库表 baidu 字段含义：
    keyword  搜索关键词（股票简称，如 "金龙鱼"）
    source   指数类型：search_all=搜索指数（整体=PC+移动），feed=资讯指数
    date     日期（日频）
    value    指数值（整数，空值记 None；资讯指数自 2017-07-03 才有数据）
"""

import datetime
import json
import logging
import os
import random
import time
import urllib.parse

import requests

from download.common import blacklist as blacklist_common
from download.common import cli
from download.common import logging as common_logging
from download.common import results
from download.common import shared
from download.common import storage
from download.common.cookies import read_cookie_interactive
from download.common.dates import build_chunks, daily_dates
from download.common.paths import DEFAULT_DB_PATH, DOWNLOAD_DIR, TEMP_DIR

try:
    from py_mini_racer import MiniRacer
except ImportError:
    MiniRacer = None

DEFAULT_COOKIE_FILE = os.path.join(DOWNLOAD_DIR, "baidu_cookie.txt")
DEFAULT_BLACKLIST_FILE = os.path.join(DOWNLOAD_DIR, "baidu_blacklist.txt")

SOURCE_NAME = "baidu"
TABLE_NAME = "baidu"
TABLE_SCHEMA = """(
    keyword TEXT,
    source TEXT,
    date DATE,
    value DOUBLE,
    PRIMARY KEY (keyword, source, date)
)"""
COLUMNS = ["keyword", "source", "date", "value"]

DEFAULT_KEYWORDS = ["金龙鱼"]
# 临时状态名前缀：C=注册制次新股、N=新股上市首日、XD=除息、XR=除权、DR=除权除息、S=未股改。
# 这些名称只在特定交易日出现，次日即恢复正常简称，百度指数基本无数据，按前缀直接跳过。
TEMP_NAME_PREFIXES = ("XD", "XR", "DR", "C", "N", "S")
SEARCH_START = "2011-01-01"   # search_all（整体=PC+移动）最早日期
FEED_START = "2017-07-03"     # 资讯指数最早日期
SEARCH_START_D = datetime.date.fromisoformat(SEARCH_START)
FEED_START_D = datetime.date.fromisoformat(FEED_START)
CHUNK_DAYS = 360              # 日频单次请求最大天数（实测 365 内返回日频），留余量
BATCH_SIZE = 1                # 单次请求最多对比关键词数
REQUEST_SLEEP_MS = 5000       # 每个请求段间隔毫秒基数（实际为 3000~5000 随机，反爬安全间隔）

ACS_JS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baidu_acs_2057.js")
ACS_JS_URL = "https://dlswbr.baidu.com/heicha/mm/2057/acs-2057.js"
SEARCH_URL = "https://index.baidu.com/api/SearchApi/index?"
FEED_URL = "https://index.baidu.com/api/FeedSearchApi/getFeedIndex?"
PTBK_URL = "https://index.baidu.com/Interface/ptbk?uniqid="
PAGE_URL = "https://index.baidu.com/v2/main/index.html"

# token 由 stub 环境算出，需与请求头 UA 保持一致（签名载荷含 UA）
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
TOKEN_TTL_S = 4 * 3600         # token 缓存时长：实测约 5.5 小时有效（起止时间戳不表有效期），留余量
BLOCK_COOLDOWN_S = 3600        # 触发风控(10001)后首次冷却 1 小时，之后逐次翻倍（2h、4h）
NIGHT_END_HOUR = 8             # 凌晨 00:00~08:00 暂停抓取
WORK_PERIOD_S = 5400           # 每连续抓取 90 分钟即休息
REST_PERIOD_S = 3600           # 休息时长 1 小时（反爬：压低单位时间请求密度）

log = logging.getLogger(SOURCE_NAME)


def refresh_acs_js(path=ACS_JS_FILE, url=ACS_JS_URL):
    """每次运行从百度 CDN 拉取最新 ACS SDK 覆盖本地缓存（国内直连，无需代理）。

    下载失败或内容异常时回退本地已有缓存；本地也没有才报错。
    """
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        # SDK 全文混淆无固定字面量，按体积做 sanity check（正常约 30KB）
        if r.content and len(r.content) > 10000:
            with open(path, "wb") as f:
                f.write(r.content)
            log.info("ACS SDK 已更新: %s（%d 字节）", path, len(r.content))
            return path
        log.warning("ACS SDK 下载内容异常（%d 字节），回退本地缓存", len(r.content or b""))
    except Exception as e:
        log.warning("ACS SDK 更新失败（%s），回退本地缓存", e)
    if os.path.exists(path):
        return path
    raise RuntimeError("ACS SDK 下载失败且本地无缓存: %s" % path)


def decrypt(ptbk, data):
    """ptbk 前一半为键、后一半为值，按映射逐字符替换。"""
    if not data:
        return ""
    n = len(ptbk) // 2
    table = dict(zip(ptbk[:n], ptbk[n:]))
    return "".join(table[c] for c in data)


class TokenProvider:
    """用 mini_racer 执行 ACS SDK 的 $BSB_2057.gs() 生成 Cipher-Text；按固定 TTL 缓存。"""

    _STUB = """
var window = {
  document: { cookie: '' },
  location: { href: '%s' },
  navigator: { userAgent: '%s' },
};
window.window = window;
""" % (PAGE_URL, USER_AGENT)

    def __init__(self, js_path=ACS_JS_FILE):
        if MiniRacer is None:
            raise ImportError("缺少 py_mini_racer，请安装：.venv/bin/pip install py_mini_racer")
        self._ctx = MiniRacer()
        self._ctx.eval(self._STUB + open(refresh_acs_js(js_path), encoding="utf-8").read())
        self._token = None
        self._made_at = 0.0

    def get(self, force=False):
        if force or self._token is None or time.time() - self._made_at > TOKEN_TTL_S:
            self._ctx.eval("window.$BSB_2057.gs(function(s, e){ window.__sign = s; });")
            token = self._ctx.eval("window.__sign")
            if not token or token == "NONE":
                raise RuntimeError("ACS SDK 未能生成 Cipher-Text token")
            self._token = token
            self._made_at = time.time()
            log.info("Cipher-Text 已生成（缓存 %d 小时，10018 自动重算）", TOKEN_TTL_S // 3600)
        return self._token


class BaiduApiClient:
    """requests 直连抓取：fetch_json 风控策略、ptbk 解码、分段抓取，产出 {search, feed, errors}。"""

    def __init__(self, cookie_text, token_provider=None, sleep_ms=REQUEST_SLEEP_MS):
        self.session = requests.Session()
        self.session.headers.update({
            "Cookie": cookie_text.strip(),
            "User-Agent": USER_AGENT,
            "Referer": PAGE_URL,
            "Accept": "application/json, text/plain, */*",
        })
        self.tokens = token_provider or TokenProvider()
        self.sleep_ms = sleep_ms
        self.block_count = 0
        self._work_started = time.monotonic()

    def _rand_sleep(self):
        time.sleep((self.sleep_ms + random.randint(0, 2000)) / 1000)

    def _pause_if_night(self):
        now = datetime.datetime.now()
        if now.hour < NIGHT_END_HOUR:
            resume = now.replace(hour=NIGHT_END_HOUR, minute=0, second=0, microsecond=0)
            wait = (resume - now).total_seconds()
            log.info("夜间暂停（%s），等待至 08:00，约 %d 分钟", now.strftime("%H:%M:%S"), round(wait / 60))
            time.sleep(max(wait, 1))
            self._work_started = time.monotonic()   # 夜间停摆不计入工作时长

    def _pause_for_work_rest(self):
        elapsed = time.monotonic() - self._work_started
        if elapsed >= WORK_PERIOD_S:
            log.info("已连续抓取 %d 分钟，休息 %d 分钟...", round(elapsed / 60), REST_PERIOD_S // 60)
            time.sleep(REST_PERIOD_S)
            self._work_started = time.monotonic()

    def fetch_json(self, url):
        r = self.session.get(url, headers={"Cipher-Text": self.tokens.get()}, timeout=60).json()
        if r and r.get("status") == 10018:
            log.warning("token 过期(10018)，重新生成")
            r = self.session.get(url, headers={"Cipher-Text": self.tokens.get(force=True)}, timeout=60).json()
        cooldown = BLOCK_COOLDOWN_S
        while r and r.get("status") == 10001:
            # 风控限流：最多冷却重试 3 次（1h -> 2h -> 4h 翻倍），第 4 次仍被限则中止脚本
            self.block_count += 1
            if self.block_count >= 4:
                raise RuntimeError("触发百度风控(10001)第 %d 次（累计已冷却重试 3 次仍被限），已中止脚本，请冷却后重跑（自动断点续抓）" % self.block_count)
            log.warning("触发风控(10001)第 %d 次，冷却 %d 小时后重试...", self.block_count, cooldown // 3600)
            time.sleep(cooldown)
            cooldown *= 2
            r = self.session.get(url, headers={"Cipher-Text": self.tokens.get()}, timeout=60).json()
        return r

    def get_ptbk(self, uniqid):
        r = self.fetch_json(PTBK_URL + uniqid)
        if r.get("status") != 0 or not r.get("data"):
            raise RuntimeError("ptbk status=%s" % r.get("status"))
        return r["data"]

    @staticmethod
    def _word_param(keywords):
        return urllib.parse.quote(json.dumps([[{"name": k, "wordType": 1}] for k in keywords]))

    def _chunk_url(self, base_url, keywords, start, end):
        return "%sarea=0&word=%s&startDate=%s&endDate=%s" % (
            base_url, self._word_param(keywords), start, end)

    def run_chunks(self, kind, base_url, keywords, chunks, extract, errors):
        out = {}
        bad_words = set()
        isolate_sleep = max(0.3, self.sleep_ms / 2000)
        for s, e in chunks:
            self._pause_if_night()
            self._pause_for_work_rest()
            tag = "%s~%s" % (s, e)
            try:
                r = self.fetch_json(self._chunk_url(base_url, keywords, s, e))
            except Exception as err:
                if "已中止" in str(err):
                    raise
                errors.append({"kind": kind, "keywords": keywords, "chunk": tag, "message": str(err)})
                self._rand_sleep()
                continue
            if r.get("status") == 0:
                try:
                    extract(r, out)
                except Exception as err:
                    if "已中止" in str(err):
                        raise
                    errors.append({"kind": kind, "keywords": keywords, "chunk": tag, "message": str(err)})
            elif len(keywords) > 1:
                # 多词整批失败时逐词隔离重试，定位未收录词；未收录词后续时间段直接跳过
                for kw in keywords:
                    if kw in bad_words:
                        continue
                    try:
                        r1 = self.fetch_json(self._chunk_url(base_url, [kw], s, e))
                        if r1.get("status") != 0:
                            errors.append({"kind": kind, "keywords": [kw], "chunk": tag,
                                           "status": r1.get("status"), "message": r1.get("message", "")})
                            if r1.get("status") != 10001:
                                bad_words.add(kw)
                        else:
                            try:
                                extract(r1, out)
                            except Exception as err:
                                if "已中止" in str(err):
                                    raise
                                errors.append({"kind": kind, "keywords": [kw], "chunk": tag, "message": str(err)})
                    except Exception as err:
                        if "已中止" in str(err):
                            raise
                        errors.append({"kind": kind, "keywords": [kw], "chunk": tag, "message": str(err)})
                    time.sleep(isolate_sleep)
            else:
                errors.append({"kind": kind, "keywords": keywords, "chunk": tag,
                               "status": r.get("status"), "message": r.get("message", "")})
            self._rand_sleep()
        return out

    def scrape_batch(self, keywords, search_chunks, feed_chunks):
        """执行一个批次，返回 {search, feed, errors}。"""
        errors = []

        def extract_search(r, out):
            ptbk = self.get_ptbk(r["data"]["uniqid"])
            for ui in r["data"].get("userIndexes") or []:
                kw = "".join(x["name"] for x in ui.get("word") or [])
                out.setdefault(kw, []).append({
                    "start": ui["all"]["startDate"], "end": ui["all"]["endDate"],
                    "values": decrypt(ptbk, ui["all"]["data"]).split(","),
                })

        def extract_feed(r, out):
            ptbk = self.get_ptbk(r["data"]["uniqid"])
            for it in r["data"].get("index") or []:
                kw = "".join(x["name"] for x in it.get("key") or [])
                out.setdefault(kw, []).append({
                    "start": it["startDate"], "end": it["endDate"],
                    "values": decrypt(ptbk, it["data"]).split(","),
                })

        search = self.run_chunks("search", SEARCH_URL, keywords, search_chunks, extract_search, errors)
        feed = self.run_chunks("feed", FEED_URL, keywords, feed_chunks, extract_feed, errors)
        return {"search": search, "feed": feed, "errors": errors}


def scrape_batches(batch_args, cookie_text, on_batch=None):
    """顺序执行所有批次；每批完成回调 on_batch(i, arg, payload)。"""
    client = BaiduApiClient(cookie_text)
    for i, arg in enumerate(batch_args):
        client.sleep_ms = arg.get("sleepMs") or REQUEST_SLEEP_MS
        payload = client.scrape_batch(arg["keywords"], arg.get("searchChunks") or [], arg.get("feedChunks") or [])
        if on_batch:
            on_batch(i, arg, payload)


def _normalize_name(name):
    """去掉所有空白字符（含全角空格），用于黑名单匹配时消除 '七 匹 狼'/'七匹狼' 之类差异。"""
    return "".join(str(name).split())


def load_blacklist(path=DEFAULT_BLACKLIST_FILE):
    """从黑名单文件读取股票简称集合（一行一个，# 开头为注释，匹配时忽略空白去重）。文件不存在返回空集。"""
    return blacklist_common.load(path, key_func=_normalize_name)


def add_to_blacklist(names, path=DEFAULT_BLACKLIST_FILE):
    """把未收录关键词追加到黑名单文件（按规范化简称去重，已在文件中的跳过），返回本次新增写入的词数。"""
    return blacklist_common.add(names, path, key_func=_normalize_name)


def load_last_dates(db_path, stage_path=None, log=None):
    """{keyword: {source: 库内最大日期}}；合并主库与未合并暂存库水位；表不存在返回空（即全量）。

    log 给定时通过公共模块打印最后日期分布（统一输出）。
    """
    rows = shared.load_max_dates(
        db_path, SOURCE_NAME, TABLE_NAME, ["keyword", "source"], "date",
        stage_path=stage_path,
    )
    out = {}
    for kw, src, d in rows:
        out.setdefault(kw, {})[src] = d
    if log is not None:
        shared.log_last_dates_distribution(log, out, TABLE_NAME, unit="个序列")
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
        sc = build_chunks(ss.isoformat(), search_end_d.isoformat(), CHUNK_DAYS) if ss <= search_end_d else []
        fc = build_chunks(fs.isoformat(), feed_end_d.isoformat(), CHUNK_DAYS) if fs <= feed_end_d else []
        for i in range(0, len(kws), batch_size):
            batches.append((kws[i:i + batch_size], sc, fc))
    return batches, skipped


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


merge_stages = storage.make_merge_stages(TABLE_NAME, TABLE_SCHEMA, COLUMNS, DEFAULT_DB_PATH)


def run(keywords=None, start=None, end=None, cookie_file=DEFAULT_COOKIE_FILE, db_path=None,
        from_stock_list=False, batch_size=BATCH_SIZE,
        sleep_ms=REQUEST_SLEEP_MS, stage_path=None, exclude_st=True,
        blacklist_file=DEFAULT_BLACKLIST_FILE, exclude_prefix=True, sort_by_lag=True, run_id=""):
    """执行百度指数抓取，写入暂存库（或主库），不在此合并。

    参数：
        keywords: 关键词列表，默认 ["金龙鱼"]
        start: 起始日期 YYYY-MM-DD；不传则按库内最大日期增量（无记录即 2011 起全量）
        end: 结束日期 YYYY-MM-DD；不传则按数据滞后规则（search 截至昨日、feed 截至前日）
        cookie_file: Cookie 文件路径，None 则交互式输入
        db_path: 主库路径，默认 download/autots.duckdb
        from_stock_list: 从 stock_list 表读取全部股票简称作为关键词（按总市值从大到小排序）
        batch_size: 单次请求对比关键词数上限，默认 1
        sleep_ms: 每个请求段间隔毫秒，默认 3000（反爬安全间隔）
        stage_path: 暂存 DuckDB 路径；"auto" 自动生成，None 直入主库；增量水位合并主库与暂存库
        exclude_st: 排除名称含 ST 的股票（默认 True）
        blacklist_file: 黑名单文件路径（一行一个股票简称，匹配时忽略空白）
        exclude_prefix: 排除 C/N/XD/XR/DR/S 开头的临时状态名（默认 True）
        sort_by_lag: 按与主库数据差距时间排序抓取顺序，差距越大越靠前（默认 True）
    返回：
        RunResult（fetch_status/rows_staged/failed/skipped 等）
    """
    log_ = common_logging.get_logger(SOURCE_NAME, run_id)
    db_path = db_path or DEFAULT_DB_PATH
    res = results.RunResult(SOURCE_NAME, run_id)

    if from_stock_list:
        keywords = shared.load_stock_names_by_total_mv(db_path)
        log_.info("从 stock_list 读取 %d 个股票简称（按总市值从大到小排序）-> %s", len(keywords), db_path)
    if keywords is None:
        keywords = DEFAULT_KEYWORDS
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]
    keywords = list(dict.fromkeys(keywords))
    if exclude_st:
        before = len(keywords)
        keywords = [k for k in keywords if not shared.is_st_name(k)]
        log_.info("排除 ST 股 %d 个，剩余 %d", before - len(keywords), len(keywords))
    blacklist = load_blacklist(blacklist_file)
    if blacklist:
        before = len(keywords)
        keywords = [k for k in keywords if _normalize_name(k) not in blacklist]
        log_.info("排除黑名单 %d 个，剩余 %d", before - len(keywords), len(keywords))
    if exclude_prefix:
        before = len(keywords)
        keywords = [k for k in keywords if not k.upper().startswith(TEMP_NAME_PREFIXES)]
        log_.info("排除临时前缀名(C/N/XD/XR/DR/S) %d 个，剩余 %d", before - len(keywords), len(keywords))
    if not keywords:
        raise ValueError("未提供关键词")

    stage_path = storage.make_stage_path(SOURCE_NAME, TEMP_DIR, stage_path)
    if stage_path:
        log_.info("暂存模式：每批写入 %s，事后用 merge 合并入主库", stage_path)
    res.stage_path = stage_path
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
        sc = build_chunks(start_d.isoformat(), search_end_d.isoformat(), CHUNK_DAYS) if start_d <= search_end_d else []
        feed_start = max(start_d, FEED_START_D)
        fc = build_chunks(feed_start.isoformat(), feed_end_d.isoformat(), CHUNK_DAYS) if feed_start <= feed_end_d else []
        batches = [(keywords[i:i + batch_size], sc, fc) for i in range(0, len(keywords), batch_size)]
        skipped = []
    else:
        last = load_last_dates(db_path, stage_path=stage_path, log=log_)
        batches, skipped = plan_batches(keywords, search_end_d, feed_end_d, last, batch_size, sort_by_lag=sort_by_lag)

    if not batches:
        log_.info("%d 个关键词库内均已覆盖到结束日（search=%s, feed=%s），无需抓取",
                  len(skipped), search_end_d, feed_end_d)
        res.fetch_status = results.FETCH_NOOP
        res.skipped = len(skipped)
        return res.finish()

    if cookie_file:
        with open(cookie_file, encoding="utf-8") as f:
            cookie_text = f.read()
    else:
        cookie_text = read_cookie_interactive()

    total = len(batches)
    log_.info("关键词 %d 个，共 %d 批（每批≤%d 词），结束日 search=%s / feed=%s",
              len(keywords), total, batch_size, search_end_d, feed_end_d)
    if skipped:
        log_.info("%d 个已是最新，跳过", len(skipped))

    stats = {}
    failed = {}
    unindexed = set()   # 本次已识别并写入黑名单的未收录词（去重）
    total_rows = [0]

    def on_batch(i, arg, payload):
        rows = payload_to_rows(payload)
        n = storage.ingest(ingest_db, TABLE_NAME, TABLE_SCHEMA, COLUMNS, rows)
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
            log_.warning("未收录 %d 词，加入黑名单（本次新增 %d）: %s",
                         len(batch_unindexed), added, "、".join(batch_unindexed))
        log_.info("批 %d/%d [%s] 入库 %d 行，累计 %d",
                  i + 1, total, "、".join(arg["keywords"]), n, total_rows[0])

    batch_args = [{"keywords": kws, "searchChunks": sc, "feedChunks": fc, "sleepMs": sleep_ms}
                  for kws, sc, fc in batches]
    try:
        scrape_batches(batch_args, cookie_text, on_batch=on_batch)
    except Exception as e:
        log_.exception("抓取异常: %s", e)
        res.error = str(e)
        res.fetch_status = results.FETCH_FAILED

    if unindexed:
        log_.warning("本次共识别 %d 个未收录关键词，已写入黑名单 %s", len(unindexed), blacklist_file)

    if failed:
        failed_map = {kw: "; ".join(sorted(msgs)) for kw, msgs in failed.items()}
        res.failed_path = results.failed_path_for(SOURCE_NAME, TEMP_DIR)
        results.write_failed_list(failed_map, res.failed_path)
        log_.warning("%d 个关键词失败（风控/token/异常，下轮重试）-> %s", len(failed), res.failed_path)

    log_.info("完成：共入库 %d 行，覆盖 %d 个序列", total_rows[0], len(stats))

    res.rows_staged = total_rows[0]
    res.success = len(stats)
    res.failed = len(failed)
    res.skipped = len(skipped)
    res.detail = {"series": stats, "unindexed": len(unindexed)}
    if res.fetch_status != results.FETCH_FAILED:
        res.fetch_status = results.FETCH_PARTIAL if failed else results.FETCH_OK
    return res.finish()


def _add_args(parser):
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
    parser.add_argument("--stage", default=None, help="暂存 DuckDB 路径；auto 自动生成，不设则直入主库")
    parser.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD（不传则按库内最大日期增量，无记录全量）")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--cookie-file", default=DEFAULT_COOKIE_FILE, help=f"从文件读取 Cookie（默认 {DEFAULT_COOKIE_FILE}）")
    parser.add_argument("--headless", action="store_true",
                        help="（已废弃，纯 API 版无需浏览器，仅为兼容旧命令保留，传了也忽略）")


def _build_kwargs(args, db_path):
    return dict(
        keywords=args.keywords,
        start=args.start,
        end=args.end,
        cookie_file=args.cookie_file,
        db_path=db_path,
        from_stock_list=args.from_stock_list,
        batch_size=args.batch_size,
        sleep_ms=args.sleep_ms,
        stage_path=args.stage,
        exclude_st=not args.include_st,
        blacklist_file=args.blacklist_file,
        exclude_prefix=not args.no_exclude_prefix,
        sort_by_lag=not args.no_sort_by_lag,
    )


def _log_start(log_, args, db_path):
    log_.info("开始（纯 API 版，无浏览器），主库: %s，排除 ST: %s，批大小 %s，间隔 %sms，按差距排序: %s",
              db_path, not args.include_st, args.batch_size, args.sleep_ms,
              not args.no_sort_by_lag)


def main():
    cli.standard_main(
        source_name=SOURCE_NAME,
        description="百度指数抓取入库（纯 API 版：mini_racer 算 token + requests 直连，search_all + feed，日频）",
        add_args=_add_args,
        kwargs_builder=_build_kwargs,
        run_func=run,
        merge_func=merge_stages,
        default_db_path=DEFAULT_DB_PATH,
        temp_dir=TEMP_DIR,
        log_start=_log_start,
    )


if __name__ == "__main__":
    main()
