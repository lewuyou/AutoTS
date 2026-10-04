# -*- coding: utf-8 -*-
"""增量窗口计算：增量起始日。

统一各数据源"起点 = max(数据源最早日, 库内最大日期+1, 用户下限)"的续抓规则。
水位读取与主库/暂存库合并见 download.common.storage。
"""

import datetime


def incremental_start(source_start, last_date=None, lower_bound=None):
    """增量起点 = max(数据源最早日, 库内最大日期+1, 用户下限)。

    source_start 为数据源允许的最早日期（如百度 search 自 2011-01-01）；
    last_date 为库内该维度已有最大日期，可为 None；
    lower_bound 为用户显式 --start 下限，可为 None。
    三者均为 datetime.date，返回 datetime.date。
    """
    candidates = [source_start]
    if last_date is not None:
        candidates.append(last_date + datetime.timedelta(days=1))
    if lower_bound is not None:
        candidates.append(lower_bound)
    return max(candidates)
