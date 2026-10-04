# -*- coding: utf-8 -*-
"""通用异常类型，供各数据源区分"空数据"与"真实失败"。"""


class EmptyDataError(RuntimeError):
    """接口返回空数据（区间内无交易日或品种无数据）。"""
