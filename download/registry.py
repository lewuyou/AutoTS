# -*- coding: utf-8 -*-
"""数据源注册表与 daily 任务定义。

新增每日抓取脚本时：
1. 在 download/sources/ 下实现模块（契约：TABLE_NAME/TABLE_SCHEMA/COLUMNS/run/merge_stages/main）
2. 在 SOURCES 注册一项 {add_args, daily_kwargs}：声明 CLI 参数与 daily 默认参数
3. 需要每日运行则把 name 加入 DAILY_ORDER（顺序即执行顺序）；
   手动定期触发的源（如 stock_list 快照表、nbjb 季报）只注册不进 DAILY_ORDER
"""

import importlib


def load_source(name):
    """惰性导入 download.sources.<name>。"""
    return importlib.import_module(f"download.sources.{name}")


def _akshare_add_args(parser):
    parser.add_argument("--adjust", default="hfq", choices=["", "qfq", "hfq"],
                        help="AKShare 复权方式（默认 hfq）")
    parser.add_argument("--akshare-delay", type=float, default=1.0,
                        help="AKShare 每股间隔秒数，默认 1.0")
    parser.add_argument("--akshare-include-st", action="store_true",
                        help="AKShare 不排除 ST 股（默认排除）")


def _akshare_daily_kwargs(args):
    return {
        "adjust": args.adjust,
        "delay": args.akshare_delay,
        "exclude_st": not args.akshare_include_st,
        "end": args.end,
        "stage_path": "auto",
        "db_path": args.db,
    }


def _baidu_add_args(parser):
    from download.sources import baidu as mod
    parser.add_argument("--baidu-keywords", default=None,
                        help="逗号分隔关键词；不传则从 stock_list 读全部股票简称")
    parser.add_argument("--baidu-cookie-file", default=None,
                        help=f"百度 Cookie 文件（默认 {mod.DEFAULT_COOKIE_FILE}）")
    parser.add_argument("--headless", action="store_true",
                        help="（已废弃，百度指数已改纯 API 无需浏览器，仅为兼容旧 cron 保留，传了也忽略）")
    parser.add_argument("--baidu-batch-size", type=int, default=None,
                        help=f"百度单次请求对比关键词数（默认 {mod.BATCH_SIZE}）")
    parser.add_argument("--baidu-sleep-ms", type=int, default=None,
                        help=f"百度每个请求段间隔毫秒（默认 {mod.REQUEST_SLEEP_MS}）")
    parser.add_argument("--baidu-include-st", action="store_true",
                        help="百度指数不排除 ST 股（默认排除）")
    parser.add_argument("--baidu-blacklist-file", default=None,
                        help=f"百度黑名单文件（默认 {mod.DEFAULT_BLACKLIST_FILE}）")


def _baidu_daily_kwargs(args):
    from download.sources import baidu as mod
    return {
        "keywords": args.baidu_keywords,
        "from_stock_list": not args.baidu_keywords,
        "cookie_file": args.baidu_cookie_file or mod.DEFAULT_COOKIE_FILE,
        "batch_size": args.baidu_batch_size or mod.BATCH_SIZE,
        "sleep_ms": args.baidu_sleep_ms or mod.REQUEST_SLEEP_MS,
        "exclude_st": not args.baidu_include_st,
        "blacklist_file": args.baidu_blacklist_file or mod.DEFAULT_BLACKLIST_FILE,
        "end": args.end,
        "stage_path": "auto",
        "db_path": args.db,
    }


def _rzrq_add_args(parser):
    from download.sources import rzrq as mod
    parser.add_argument("--rzrq-delay", type=float, default=mod.DEFAULT_DELAY,
                        help=f"两融每股间隔秒数，默认 {mod.DEFAULT_DELAY}")
    parser.add_argument("--rzrq-include-st", action="store_true",
                        help="两融不排除 ST 股（默认排除）")


def _rzrq_daily_kwargs(args):
    return {
        "codes": None,
        "end": args.end,
        "db_path": args.db,
        "delay": args.rzrq_delay,
        "exclude_st": not args.rzrq_include_st,
        "stage_path": "auto",
    }


def _nbjb_add_args(parser):
    from download.sources import nbjb as mod
    parser.add_argument("--nbjb-delay", type=float, default=mod.DEFAULT_DELAY,
                        help=f"业绩报告每股间隔秒数，默认 {mod.DEFAULT_DELAY}")
    parser.add_argument("--nbjb-include-st", action="store_true",
                        help="业绩报告不排除 ST 股（默认排除）")


def _nbjb_daily_kwargs(args):
    return {
        "codes": None,
        "end": args.end,
        "db_path": args.db,
        "delay": args.nbjb_delay,
        "exclude_st": not args.nbjb_include_st,
        "stage_path": "auto",
    }


def _guzhi_add_args(parser):
    from download.sources import guzhi as mod
    parser.add_argument("--guzhi-delay", type=float, default=mod.DEFAULT_DELAY,
                        help=f"估值每股间隔秒数，默认 {mod.DEFAULT_DELAY}")
    parser.add_argument("--guzhi-include-st", action="store_true",
                        help="估值不排除 ST 股（默认排除）")


def _guzhi_daily_kwargs(args):
    return {
        "codes": None,
        "end": args.end,
        "db_path": args.db,
        "delay": args.guzhi_delay,
        "exclude_st": not args.guzhi_include_st,
        "stage_path": "auto",
    }


def _holiday_add_args(parser):
    pass  # 节假日无需额外参数，daily 默认抓当年全年（受 --end 截断）
def _holiday_daily_kwargs(args):
    return {
        "end": args.end,
        "db_path": args.db,
        "stage_path": "auto",
    }


def _stock_list_add_args(parser):
    pass  # 快照表无需额外参数，全量重建；不进 DAILY_ORDER，仅手动 --sources stock_list 触发
def _stock_list_daily_kwargs(args):
    return {
        "db_path": args.db,
        "stage_path": "auto",
    }


def _tv_macro_add_args(parser):
    parser.add_argument("--tv-macro-symbols", default=None,
                        help="逗号分隔 TradingView 宏观品种代码（默认读 tradingview_macro_symbols 配置表）")
    parser.add_argument("--tv-macro-interval", default="1D",
                        help="TradingView 宏观 K 线周期（默认 1D）")
    parser.add_argument("--tv-macro-no-proxy", action="store_true",
                        help="TradingView 宏观不走本机 7897 代理（默认走代理）")


def _tv_macro_daily_kwargs(args):
    return {
        "symbols": args.tv_macro_symbols,
        "interval": args.tv_macro_interval,
        "use_proxy": not args.tv_macro_no_proxy,
        "end": args.end,
        "db_path": args.db,
        "stage_path": "auto",
    }


SOURCES = {
    "akshare": {"add_args": _akshare_add_args, "daily_kwargs": _akshare_daily_kwargs},
    "baidu": {"add_args": _baidu_add_args, "daily_kwargs": _baidu_daily_kwargs},
    "rzrq": {"add_args": _rzrq_add_args, "daily_kwargs": _rzrq_daily_kwargs},
    "nbjb": {"add_args": _nbjb_add_args, "daily_kwargs": _nbjb_daily_kwargs},
    "guzhi": {"add_args": _guzhi_add_args, "daily_kwargs": _guzhi_daily_kwargs},
    "holiday": {"add_args": _holiday_add_args, "daily_kwargs": _holiday_daily_kwargs},
    "stock_list": {"add_args": _stock_list_add_args, "daily_kwargs": _stock_list_daily_kwargs},
    "tv_macro": {"add_args": _tv_macro_add_args, "daily_kwargs": _tv_macro_daily_kwargs},
}
DAILY_ORDER = ["holiday", "akshare", "baidu", "rzrq", "guzhi", "tv_macro"]  # holiday 先行串行（交易日历供其他源判断已齐全交易日），其余源并行抓取、按此顺序串行合并；nbjb 季频不进 DAILY_ORDER，仅手动 --sources nbjb 触发


def add_source_args(parser):
    """把已注册数据源的 CLI 参数加到 parser。"""
    for name in SOURCES:
        SOURCES[name]["add_args"](parser)
