from __future__ import absolute_import
import numpy as np
from pyti import catch_errors
from pyti.function_helper import fill_for_noncomputable_vals
from six.moves import range


def conversion_base_line_helper(data, period):
    """
    The only real difference between TenkanSen and KijunSen is the period value
    """
    catch_errors.check_for_period_error(data, period)
    cblh = [(np.max(data[idx+1-period:idx+1]) +
            np.min(data[idx+1-period:idx+1])) / 2 for idx in range(period-1, len(data))]

    cblh = fill_for_noncomputable_vals(data, cblh)
    return cblh


def tenkansen(data, period=9):
    """
    TenkanSen (Conversion Line)

    Formula:
    (H + L) / 2  :: default period=9
    """
    ts = conversion_base_line_helper(data, period)
    return ts


def kijunsen(data, period=26):
    """
    KijunSen (Base Line)

    Formula:
    (H + L) / 2  :: default period=26
    """
    ks = conversion_base_line_helper(data, period)
    return ks


def chiku_span(data):
    """
    Chiku Span (Lagging Span)

    Formula:
    Close shifted back 26 bars
    """
    cs = data[25::]
    nan_padding = np.full(25, np.nan)  # 创建一个长度为 25 的 NaN 数组
    cs = np.concatenate([nan_padding, cs])  # 将 NaN 数组和计算结果合并
    return cs


def senkou_a(data):
    """
    Senkou A (Leading Span A)

    Formula:
    (TenkanSen + KijunSen) / 2 :: Shift Forward 26 bars
    """
    sa = (tenkansen(data) + kijunsen(data)) / 2
    # shift forward with NaN padding
    shift_by = np.repeat(np.nan, 26)
    sa = np.append(shift_by, sa)
    # 裁剪数组以保持原始长度不变
    sa = sa[:len(data)]
    return sa


def senkou_b(data, period=52):
    """
    Senkou B (Leading Span B)

    Formula:
    (H + L) / 2  :: default period=52 :: shifted forward 26 bars
    """
    sb = conversion_base_line_helper(data, period)
    # shift forward with NaN padding
    shift_by = np.repeat(np.nan, 26)
    sb = np.append(shift_by, sb)
    # 裁剪数组以保持原始长度不变
    sb = sb[:len(data)]
    return sb
