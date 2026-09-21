# -*- coding: utf-8 -*-
"""抓取 TradingView 布局指标的加密 text，存成 tv_study_texts.json。

背景：
    tradingview_study.py 用 create_study 拉指标数据时，参数里的 text 字段是
    TradingView 对脚本身份的端到端加密签名。该 text 对同一指标长期固定、可复用
    （已验证跨会话一致、换 symbol 有效），但**无法凭空构造**——必须从你登录的
    浏览器建立 WebSocket 会话时抓一次。本脚本自动化这个过程。

何时运行：
    * 新增了一个布局（图表编号），里面有新指标，需要抓它们的 text
    * tradingview_study.py 跑指标时报 "study_error" / 抓不到数据，怀疑 text 失效
    * 布局里指标的版本升级了（metaInfo 里 [v.xx] 变了）

用法：
    python -m AutoTS.download.tv.fetch_tv_study_texts --layout dP9MRLfC
    python -m AutoTS.download.tv.fetch_tv_study_texts --layout dP9MRLfC --cookie-file tv_cookie.txt
    python -m AutoTS.download.tv.fetch_tv_study_texts --layout dP9MRLfC --out tv_study_texts.json --headless

流程：
    1. 读 cookie（tv_cookie.txt，需含 sessionid，登录态）
    2. 用 Playwright 打开布局页面（带 cookie + 代理），等图表加载
    3. 拦截该页面的 WebSocket 帧，提取所有 create_study 消息里的加密 text
    4. 与布局配置（initData.content 里的 metaInfo/inputs）按 study_id 合并
    5. 写入 tv_study_texts.json（已存在的 study_id 会被新值覆盖，其余的保留）

依赖：
    pip install playwright
    playwright install chromium

注意：
    * 这个脚本是一次性手动运行的（text 失效或换布局时才跑），不是日常抓数流程的一部分。
    * 需要本机 7897 代理（可用 --no-proxy 关闭）。
    * 布局页面必须是你账号有权限访问的（私有/收费指标需你有相应权限）。
"""

import argparse
import json
import os
import re
import sys

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_COOKIE_FILE = os.path.join(_DIR, "tv_cookie.txt")
DEFAULT_OUT_FILE = os.path.join(_DIR, "tv_study_texts.json")
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7897


def parse_cookie_text(text):
    """把 cookie 字符串解析成 playwright 需要的 cookie 列表。"""
    text = (text or "").strip()
    pairs = {}
    for part in text.replace(";", "\n").split("\n"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        if k and k.lower() != "undefined":
            pairs[k] = v.strip()
    return [{"name": k, "value": v, "domain": ".tradingview.com", "path": "/"}
            for k, v in pairs.items()]


def fetch_texts(layout_id, cookie_file=DEFAULT_COOKIE_FILE, out_file=DEFAULT_OUT_FILE,
                headless=True, use_proxy=True, wait_seconds=20):
    """打开布局页面抓 create_study 的加密 text，合并布局配置后写入 out_file。"""
    if sync_playwright is None:
        raise ImportError("缺少 playwright，请安装：python3 -m pip install playwright && playwright install chromium")

    with open(cookie_file, encoding="utf-8") as f:
        cookies = parse_cookie_text(f.read())
    if not any(c["name"] == "sessionid" for c in cookies):
        raise ValueError("cookie 文件里没有 sessionid，请先登录 TradingView 再导出 cookie")

    # 收集 WebSocket 帧里的 create_study
    collected = {}  # study_id -> {text, pineId, pineVersion}

    def on_websocket(ws):
        def on_frame(payload):
            if isinstance(payload, bytes):
                try:
                    payload = payload.decode("utf-8", "ignore")
                except Exception:
                    return
            if "create_study" not in payload:
                return
            for m in re.findall(r"~m~\d+~m~(\{.*?\})(?=~m~|$)", payload, re.S):
                try:
                    obj = json.loads(m)
                except Exception:
                    continue
                if obj.get("m") != "create_study":
                    continue
                p = obj.get("p", [])
                if len(p) < 6:
                    continue
                sid = p[1]
                param = p[5]
                if isinstance(param, dict) and "text" in param:
                    collected[sid] = {
                        "text": param["text"],
                        "pineId": param.get("pineId", ""),
                        "pineVersion": param.get("pineVersion", ""),
                    }
        ws.on("framereceived", on_frame)
        ws.on("framesent", on_frame)

    url = f"https://cn.tradingview.com/chart/{layout_id}/"
    print(f"[抓text] 打开布局 {url} ...")
    with sync_playwright() as pw:
        launch_kwargs = {"headless": headless}
        if use_proxy:
            launch_kwargs["proxy"] = {"server": f"http://{PROXY_HOST}:{PROXY_PORT}"}
        browser = pw.chromium.launch(**launch_kwargs)
        context = browser.new_context()
        context.add_cookies(cookies)
        page = context.new_page()
        page.on("websocket", on_websocket)
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        # 等图表加载、所有 study 的 create_study 发出
        print(f"[抓text] 等待 {wait_seconds}s 让所有指标加载 ...")
        page.wait_for_timeout(wait_seconds * 1000)
        browser.close()

    if not collected:
        raise RuntimeError("没抓到任何 create_study 的 text。可能是：页面没加载完（加大 --wait）、cookie 失效、或布局里没指标。")
    print(f"[抓text] 抓到 {len(collected)} 个指标的加密 text")

    # 合并布局配置里的 metaInfo（从布局 HTML 提取，便于阅读）
    try:
        from AutoTS.download.tv import tradingview_study as ts
        cookie_str = open(cookie_file, encoding="utf-8").read().strip()
        _, content = ts.fetch_layout_page(layout_id, cookie_str, use_proxy=use_proxy)
        meta = {}
        for pane in content.get("charts", [{}])[0].get("panes", []):
            for s in pane.get("sources", []):
                if s.get("type") != "MainSeries":
                    meta[s.get("id")] = s.get("metaInfo", "")
        for sid in collected:
            collected[sid]["metaInfo"] = meta.get(sid, "")
    except Exception as e:
        print(f"[抓text] 警告：合并 metaInfo 失败（不影响 text 使用）: {e}", file=sys.stderr)

    # 写入：保留旧文件里其他 study，覆盖本次抓到的
    existing = {}
    if os.path.exists(out_file):
        with open(out_file, encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(collected)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"[抓text] 已写入 {out_file}（共 {len(existing)} 个指标）")
    for sid, info in collected.items():
        print(f"  {sid}: {info.get('metaInfo', '')[:50]}  text_len={len(info['text'])}")
    return existing


def main():
    parser = argparse.ArgumentParser(description="抓取 TradingView 布局指标的加密 text")
    parser.add_argument("--layout", required=True, help="布局编号（图表 URL 里的那段，如 dP9MRLfC）")
    parser.add_argument("--cookie-file", default=DEFAULT_COOKIE_FILE, help="cookie 文件（默认 download/tv/tv_cookie.txt）")
    parser.add_argument("--out", default=DEFAULT_OUT_FILE, help="输出文件（默认 download/tv/tv_study_texts.json）")
    parser.add_argument("--wait", type=int, default=20, help="等图表加载的秒数（默认 20，指标多/网络慢可加大）")
    parser.add_argument("--headless", action="store_true", help="无头模式（默认有头，便于观察是否加载成功）")
    parser.add_argument("--no-proxy", action="store_true", help="不走本机 7897 代理")
    args = parser.parse_args()

    fetch_texts(
        layout_id=args.layout,
        cookie_file=args.cookie_file,
        out_file=args.out,
        headless=args.headless,
        use_proxy=not args.no_proxy,
        wait_seconds=args.wait,
    )


if __name__ == "__main__":
    main()
