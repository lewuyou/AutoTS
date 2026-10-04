# -*- coding: utf-8 -*-
"""抓取黑名单文件读写：一行一个条目，# 开头为注释，用于跳过确认无数据的股票/关键词。

条目格式 "键 附注"（附注可空，如代码后附股票名称便于阅读）。匹配键由 key_func 提取：
默认取首列（rzrq 按代码匹配，附注名称不影响）；baidu 传整行去空白（按规范化简称匹配）。
黑名单文件路径由各源模块自定义（如 download/baidu_blacklist.txt、download/rzrq_blacklist.txt）。
"""

import os


def _first_token(line):
    return line.split()[0]


def load(path, key_func=None):
    """读取黑名单文件，返回匹配键集合；路径为空或文件不存在返回空集。

    key_func: 从一行文本提取匹配键，默认取首列（忽略附注）。
    """
    keys = set()
    if not path or not os.path.exists(path):
        return keys
    key_func = key_func or _first_token
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            keys.add(key_func(line))
    return keys


def add(entries, path, key_func=None):
    """把条目追加到黑名单文件（按匹配键去重，已在文件中的跳过），返回本次新增写入的数量。

    entries: str（键本身）或 (键, 附注) 元组列表；写入 "键 附注" 行（附注为空只写键）。
    key_func: 提取匹配键用于去重，默认取首列；与 load 的 key_func 需一致。
    """
    if not path or not entries:
        return 0
    key_func = key_func or _first_token
    existing = load(path, key_func=key_func)
    new = []
    for e in entries:
        key, note = e if isinstance(e, tuple) else (e, "")
        k = key_func(str(key))
        if k in existing:
            continue
        existing.add(k)
        new.append((key, note))
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
        for key, note in new:
            f.write(f"{key} {note}\n" if note else f"{key}\n")
    return len(new)
