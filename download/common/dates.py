# -*- coding: utf-8 -*-
"""通用日期工具：日期分段、逐日序列。

baidu 等数据源按天分段抓取时用到；与具体数据源解耦，供多处复用。
"""

import datetime


def build_chunks(start, end, chunk_days):
    """把 [start, end]（YYYY-MM-DD 字符串）切成连续小段，每段跨度 <= chunk_days。

    返回 [[start_iso, end_iso], ...]（闭区间，含两端日期）。
    """
    s = datetime.date.fromisoformat(start)
    e = datetime.date.fromisoformat(end)
    chunks = []
    while s <= e:
        c_end = min(s + datetime.timedelta(days=chunk_days), e)
        chunks.append([s.isoformat(), c_end.isoformat()])
        s = c_end + datetime.timedelta(days=1)
    return chunks


def daily_dates(start, end):
    """逐日日期序列（含两端），返回 [YYYY-MM-DD, ...]。"""
    s = datetime.date.fromisoformat(start)
    e = datetime.date.fromisoformat(end)
    out = []
    d = s
    while d <= e:
        out.append(d.isoformat())
        d += datetime.timedelta(days=1)
    return out
