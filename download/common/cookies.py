# -*- coding: utf-8 -*-
"""通用 Cookie 工具：解析浏览器粘贴的 Cookie 文本、交互式读取。

baidu 等需要登录态的数据源共用；与具体站点解耦，供多处复用。
"""

import json


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
    print("请粘贴浏览器抓取的 Cookie（登录后，DevTools 里复制请求头里的 Cookie 整串）：")
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
