# -*- coding: utf-8 -*-
"""宏观经济数据抓取配置（TradingView 品种代码）。

每项：(symbol, 中文名称)。symbol 为 TradingView 完整代码（EXCHANGE:CODE 或纯代码）。
"""

# (symbol, 中文名称)
MACRO_SYMBOLS = [
    ("TVC:CN10Y", "中国10年期国债收益率"),
    ("TVC:US10Y", "美国10年期国债收益率"),
    ("SPCFD:SPX", "标普500指数"),
    ("SSE:000001", "上证综合指数"),
    ("SSE:510300", "沪深300ETF"),
    ("USDCNY", "美元人民币汇率"),
    ("ATW1!", "鹿特丹煤炭期货合约"),
    ("NG1!", "美国天然气期货"),
    ("NASDAQ:SOX", "费城半导体指数"),
    ("NASDAQ:IXIC", "纳斯达克综合指数"),
    ("NASDAQ:NDX", "纳斯达克100指数"),
    ("CBOE:VIX", "VIX恐慌指数"),
    ("HSI:HSI", "恒生指数"),
    ("TVC:DXY", "美元指数"),
    ("NYMEX:CL1!", "WTI原油期货"),
    ("COMEX:GC1!", "黄金期货"),
    ("COMEX:HG1!", "铜期货"),
]
