# -*- coding: utf-8 -*-
"""AutoTS 数据下载总入口。

按数据源分别调用 download/ 下的模块，支持单独或批量抓取。

用法：
    python -m AutoTS.download_all                          # 抓全部默认数据源（当年）
    python -m AutoTS.download_all --source baidu           # 只抓百度指数
    python -m AutoTS.download_all --source holiday         # 只抓节假日
    python -m AutoTS.download_all --source gzfx            # 只抓估值通道
    python -m AutoTS.download_all --source nbjb            # 只抓业绩报告
    python -m AutoTS.download_all --source rzrq            # 只抓融资融券
    python -m AutoTS.download_all --source tradingview     # 只抓 TradingView K线
    python -m AutoTS.download_all --source tradingview_study  # 只抓 TradingView 布局指标
    python -m AutoTS.download_all --source all             # 抓全部
    python -m AutoTS.download_all --start 2013-01-01       # 指定起始日期
    python -m AutoTS.download_all --keywords 金龙鱼,浪潮信息
    python -m AutoTS.download_all --codes 300999,688223    # 估值/业绩/两融/K线/指标股票代码
    python -m AutoTS.download_all --cookie-file c.txt      # 百度 Cookie 文件
    python -m AutoTS.download_all --tv-layout dP9MRLfC     # TradingView 指标布局编号
    python -m AutoTS.download_all --headless               # 百度无头模式

说明：
    * 每个数据源独立抓取，一个失败不影响另一个。
    * 默认只抓当年数据（日常增量）；传 --start 可抓历史。TradingView 默认抓全历史。
    * 百度指数需要 Cookie；节假日、估值通道、业绩报告、融资融券、TradingView K线不需要。
    * TradingView 指标（tradingview_study）需要 Cookie（含收费/私有指标），不在 all 里，需单独指定。
      TradingView 走本机 7897 代理（可用 --tv-no-proxy 关闭）。
"""

import argparse
import datetime
import sys

from download import baidu, gzfx, holiday, nbjb, rzrq
from download.tv import tradingview, tradingview_study


def run_source(name, args):
    """运行单个数据源，返回 (是否成功, 结果或异常)。"""
    try:
        if name == "baidu":
            result = baidu.run(
                keywords=args.keywords,
                start=args.start,
                end=args.end,
                cookie_file=args.cookie_file,
                db_path=args.baidu_db,
                headless=args.headless,
                no_csv=args.no_csv,
            )
        elif name == "holiday":
            years = [int(y.strip()) for y in args.years.split(",") if y.strip()] if args.years else None
            result = holiday.run(
                start=args.start,
                end=args.end,
                years=years,
                db_path=args.holiday_db,
                no_csv=args.no_csv,
            )
        elif name == "gzfx":
            result = gzfx.run(
                codes=args.codes,
                date_type=args.datetype,
                db_path=args.gzfx_db,
            )
        elif name == "nbjb":
            result = nbjb.run(
                codes=args.codes,
                db_path=args.nbjb_db,
            )
        elif name == "rzrq":
            result = rzrq.run(
                codes=args.codes,
                db_path=args.rzrq_db,
            )
        elif name == "tradingview":
            result = tradingview.run(
                codes=args.codes,
                start=args.start,
                end=args.end,
                cookie_file=args.tv_cookie_file,
                db_path=args.tv_db,
                use_proxy=not args.tv_no_proxy,
                no_csv=args.no_csv,
            )
        elif name == "tradingview_study":
            result = tradingview_study.run(
                codes=args.codes,
                layout_id=args.tv_layout,
                start=args.start,
                end=args.end,
                cookie_file=args.tv_cookie_file or tradingview_study.DEFAULT_COOKIE_FILE,
                db_path=args.tv_db,
                use_proxy=not args.tv_no_proxy,
            )
        else:
            raise ValueError(f"未知数据源: {name}")
        return True, result
    except Exception as e:
        return False, e


def main():
    parser = argparse.ArgumentParser(description="AutoTS 数据下载总入口")
    parser.add_argument(
        "--source",
        default="all",
        choices=["all", "baidu", "holiday", "gzfx", "nbjb", "rzrq", "tradingview", "tradingview_study"],
        help="要抓取的数据源，默认 all",
    )
    parser.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD（默认当年 1 月 1 日）")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认今天/当年 12 月 31 日）")
    parser.add_argument("--years", default=None, help="逗号分隔年份，如 2024,2025,2026（节假日优先）")
    parser.add_argument("--keywords", default=None, help="逗号分隔关键词（百度指数）")
    parser.add_argument("--codes", default=None, help="逗号分隔股票代码（估值/业绩/两融），如 300999,688223")
    parser.add_argument("--datetype", type=int, default=1, choices=[1, 2, 3, 4],
                        help="估值通道时间范围：1=近1年(日频) 2=近3年 3=近5年 4=近10年，默认 1")
    parser.add_argument("--cookie-file", default=None, help="百度 Cookie 文件路径")
    parser.add_argument("--baidu-db", default=None, help="百度指数 DuckDB 路径（默认 download/autots.duckdb）")
    parser.add_argument("--holiday-db", default=None, help="节假日 DuckDB 路径（默认 download/autots.duckdb）")
    parser.add_argument("--gzfx-db", default=None, help="估值通道 DuckDB 路径（默认 download/autots.duckdb）")
    parser.add_argument("--nbjb-db", default=None, help="业绩报告 DuckDB 路径（默认 download/autots.duckdb）")
    parser.add_argument("--rzrq-db", default=None, help="融资融券 DuckDB 路径（默认 download/autots.duckdb）")
    parser.add_argument("--tv-db", default=None, help="TradingView K线/指标 DuckDB 路径（默认 download/autots.duckdb）")
    parser.add_argument("--tv-cookie-file", default=None, help="TradingView Cookie 文件（K线可选；指标必需，默认 download/tv/tv_cookie.txt）")
    parser.add_argument("--tv-layout", default="dP9MRLfC", help="TradingView 指标布局编号，默认 dP9MRLfC")
    parser.add_argument("--tv-no-proxy", action="store_true", help="TradingView 不走本机 7897 代理")
    parser.add_argument("--headless", action="store_true", help="百度指数无头模式")
    parser.add_argument("--no-csv", action="store_true", help="不写 CSV，只入库")
    args = parser.parse_args()

    sources = ["baidu", "holiday", "gzfx", "nbjb", "rzrq", "tradingview"] if args.source == "all" else [args.source]
    results = {}
    failed = []

    for name in sources:
        print(f"\n{'='*50}")
        print(f"开始抓取数据源: {name}")
        print(f"{'='*50}")
        ok, result = run_source(name, args)
        results[name] = result
        if ok:
            print(f"\n[{name}] 完成")
        else:
            print(f"\n[{name}] 失败: {result}", file=sys.stderr)
            failed.append(name)

    print(f"\n{'='*50}")
    print("汇总")
    print(f"{'='*50}")
    for name, result in results.items():
        if name in failed:
            print(f"  {name}: 失败 - {result}")
        else:
            print(f"  {name}: 成功，入库 {result.get('rows_count', 0)} 行")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
