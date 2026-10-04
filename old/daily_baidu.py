# -*- coding: utf-8 -*-
"""每日百度指数增量更新（总入口）。

每天执行一次，把主库 baidu 表缺失的搜索指数(search_all)/资讯指数(feed)补齐：
    1. 读取主库 baidu 表，打印每个关键词（股票简称）最后日期分布，看哪些词滞后
    2. 调用 download.baidu.run(stage_path="auto") 暂存模式抓取缺失数据
       —— 默认从 stock_list 表读全部股票简称（排除 ST/黑名单/C-N-XD 等临时状态名），
          每个关键词独立计算增量起点 = max(指数收录起始日, 库内该词最大日期+1)，
          结束日按数据滞后取：search 截至昨日、feed 截至前日（不空查尚未发布的新一天）
    3. 抓取全部完成后，用 merge_stages 把暂存库统一 UPSERT 入主库
    4. 全程 stdout/stderr 同时写日志文件 download/temp/daily_baidu_<时间戳>.log

用法：
    python daily_baidu.py                        # 全部股票简称增量，暂存+合并+日志
    python daily_baidu.py --keywords 金龙鱼,浪潮信息   # 只抓指定关键词
    python daily_baidu.py --headless             # 无头模式（cron 建议加）
    python daily_baidu.py --batch-size 2 --sleep-ms 5000   # 收紧抓取节奏防风控
    python daily_baidu.py --include-st           # 不排除 ST 股
    python daily_baidu.py --no-sort-by-lag       # 关闭按主库数据差距排序（默认差距越大越靠前）
    python daily_baidu.py --cookie-file c.txt    # 指定 Cookie 文件

注意：
    * 必须提供 Cookie（默认 download/baidu_cookie.txt），否则百度指数接口无法取 token。
    * 抓取走 Playwright 浏览器（百度指数需要页面 Paris SDK 生成 Cipher-Text token）。
    * 内置夜间 00:00~08:00 自动暂停（百度夜间不更新新一天且更易触发风控），
      建议 cron 安排在 08:00~24:00（如早 09:00 或晚 18:00）。
    * 触发风控(10001)自动冷却 1 小时重试，第二次触发即中止脚本；失败关键词写入
      download/temp/baidu_failed_<时间戳>.txt，重跑即可补抓（自动断点续抓）。
    * 未收录关键词（百度指数无该词数据）自动加入黑名单 baidu_blacklist.txt，下次运行不再查询。

建议 crontab（工作日早 09:00）：
    0 9 * * 1-5  cd /Users/lwy/Code/qoder/AutoTS && .venv/bin/python daily_baidu.py --headless >> /dev/null 2>&1

入库表 baidu 字段含义：
    keyword  搜索关键词（股票简称，如 "金龙鱼"）
    source   指数类型：search_all=搜索指数（整体=PC+移动），feed=资讯指数
    date     日期（日频）
    value    指数值（整数，空值记 None；资讯指数自 2017-07-03 才有数据）
"""

import argparse
import contextlib
import datetime
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from download import baidu  # noqa: E402

TEMP_DIR = os.path.join(_SCRIPT_DIR, "download", "temp")


class _Tee:
    """同时写入多个流，用于把 print 输出同时打到控制台和日志文件。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


def print_db_state(db_path):
    """打印主库 baidu 表每个关键词最后日期分布。"""
    last = baidu.load_last_dates(db_path)
    today = datetime.date.today()
    search_end_d = today - datetime.timedelta(days=1)
    feed_end_d = today - datetime.timedelta(days=2)
    if not last:
        print("[每日百度] 主库 baidu 表无记录，本次按全量抓取（search 自 2011-01-01，feed 自 2017-07-03）")
        return
    search_dates = {k: v["search_all"] for k, v in last.items() if "search_all" in v}
    feed_dates = {k: v["feed"] for k, v in last.items() if "feed" in v}
    search_lag = sum(1 for d in search_dates.values() if d < search_end_d)
    feed_lag = sum(1 for d in feed_dates.values() if d < feed_end_d)
    print(f"[每日百度] 主库已有 {len(last)} 个关键词")
    if search_dates:
        print(f"[每日百度] search_all：{len(search_dates)} 词，最新 {max(search_dates.values()):%Y-%m-%d}，滞后待补 {search_lag} 词")
    if feed_dates:
        print(f"[每日百度] feed：{len(feed_dates)} 词，最新 {max(feed_dates.values()):%Y-%m-%d}，滞后待补 {feed_lag} 词")


def run_daily(args):
    db_path = args.db or baidu.DEFAULT_DB_PATH
    cookie_file = args.cookie_file or baidu.DEFAULT_COOKIE_FILE
    started = datetime.datetime.now()
    print("=" * 60)
    print(f"[每日百度] 开始 {started:%Y-%m-%d %H:%M:%S}")
    print(f"[每日百度] 主库: {db_path}")
    print(f"[每日百度] Cookie: {cookie_file}，无头: {args.headless}，排除 ST: {not args.include_st}，批大小 {args.batch_size}，间隔 {args.sleep_ms}ms，按差距排序: {not args.no_sort_by_lag}")
    print("=" * 60)

    if not os.path.exists(cookie_file):
        print(f"[每日百度] Cookie 文件不存在: {cookie_file}")
        print("[每日百度] 请先登录 index.baidu.com，复制请求头里的 Cookie 写入该文件，或用 --cookie-file 指定")
        return 1

    print_db_state(db_path)

    result = baidu.run(
        keywords=args.keywords,              # 为 None 时配合 from_stock_list=True 读全部股票简称
        from_stock_list=not args.keywords,
        cookie_file=cookie_file,
        db_path=db_path,
        headless=args.headless,
        no_csv=True,                          # 全量简称数据太大，不写 CSV
        stage_path="auto",                    # 暂存模式：每批先写暂存库，不碰主库
        exclude_st=not args.include_st,
        blacklist_file=args.blacklist_file or baidu.DEFAULT_BLACKLIST_FILE,
        batch_size=args.batch_size,
        sleep_ms=args.sleep_ms,
        sort_by_lag=not args.no_sort_by_lag,
    )

    stage_path = result.get("stage_path")
    if stage_path and os.path.exists(stage_path):
        print("\n[每日百度] 抓取完成，开始合并暂存库入主库 ...")
        total, nfiles = baidu.merge_stages(stage_path, db_path=db_path)
        print(f"[每日百度] 合并完成：{nfiles} 个暂存文件，共 {total} 行 -> {db_path}")
    else:
        print("\n[每日百度] 无新数据，跳过合并（主库已是最新）")

    elapsed = datetime.datetime.now() - started
    print("\n" + "=" * 60)
    print(f"[每日百度] 结束 {datetime.datetime.now():%Y-%m-%d %H:%M:%S}，耗时 {elapsed}")
    print(f"[每日百度] 本次入库 {result.get('rows_count', 0)} 行，覆盖 {len(result.get('series', {}))} 个序列")
    if result.get("failed_path"):
        print(f"[每日百度] 失败/无数据清单: {result['failed_path']}")
    print("=" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser(description="每日百度指数增量更新（暂存 + 合并 + 日志）")
    parser.add_argument("--keywords", default=None, help="逗号分隔关键词；不传则从 stock_list 读全部股票简称")
    parser.add_argument("--cookie-file", default=None, help=f"Cookie 文件路径（默认 {baidu.DEFAULT_COOKIE_FILE}）")
    parser.add_argument("--db", default=None, help="主库 DuckDB 路径（默认 download/autots.duckdb）")
    parser.add_argument("--headless", action="store_true", help="无头模式（cron 建议加）")
    parser.add_argument("--batch-size", type=int, default=baidu.BATCH_SIZE, help=f"单次请求对比关键词数，默认 {baidu.BATCH_SIZE}")
    parser.add_argument("--sleep-ms", type=int, default=baidu.REQUEST_SLEEP_MS, help=f"每个请求段间隔毫秒，默认 {baidu.REQUEST_SLEEP_MS}")
    parser.add_argument("--include-st", action="store_true", help="不排除 ST 股（默认排除）")
    parser.add_argument("--no-sort-by-lag", action="store_true", help="不按与主库数据差距时间排序（默认差距越大越靠前）")
    parser.add_argument("--blacklist-file", default=None, help=f"黑名单文件路径（默认 {baidu.DEFAULT_BLACKLIST_FILE}）")
    args = parser.parse_args()

    os.makedirs(TEMP_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(TEMP_DIR, f"daily_baidu_{stamp}.log")

    with open(log_path, "w", encoding="utf-8") as log_f:
        out_tee = _Tee(sys.stdout, log_f)
        err_tee = _Tee(sys.stderr, log_f)
        with contextlib.redirect_stdout(out_tee), contextlib.redirect_stderr(err_tee):
            try:
                code = run_daily(args)
            except Exception as e:
                print(f"\n[每日百度] 失败: {e}", file=sys.stderr)
                code = 1

    print(f"日志已保存: {log_path}")
    sys.exit(code)


if __name__ == "__main__":
    main()
