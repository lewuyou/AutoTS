# -*- coding: utf-8 -*-
"""每日 A 股行情增量更新（总入口）。

每天执行一次，把主库 akshare_tx 缺失的行情补齐：
    1. 读取主库 akshare_tx，打印每只股票最后日期分布（看哪些股滞后）
    2. 调用 download.akshare.run(stage_path="auto") 暂存模式抓取缺失数据
       —— 每股增量起点 = max(list_date, 主库内该股同复权方式最大日期 + 1)，
          首次全量与每日增量同一条路径，UPSERT 可重跑、可断点续抓
    3. 抓取全部完成后，用 merge_stages 把暂存库统一 UPSERT 入主库
    4. 全程 stdout/stderr 同时写日志文件 download/temp/daily_akshare_<时间戳>.log，
       控制台实时看进度，事后可回看日志

用法：
    python daily_akshare.py                     # 默认：后复权，排除 ST，每股间隔 1s
    python daily_akshare.py --adjust qfq        # 前复权
    python daily_akshare.py --delay 2.0         # 每股间隔秒数（防腾讯限流）
    python daily_akshare.py --include-st        # 不排除 ST 股
    python daily_akshare.py --db 某路径.duckdb   # 指定主库（默认 download/autots.duckdb）

建议 crontab（交易日早间跑）：
    0 18 * * 1-5  cd /Users/lwy/Code/qoder/AutoTS && .venv/bin/python daily_akshare.py >> /dev/null 2>&1

入库表 akshare_tx 字段含义（结构定义见 download/akshare.py）：
    symbol      股票代码（6 位）
    date        交易日
    open/close/high/low  开/收/高/低价（元）
    volume      成交量（股；腾讯接口原始单位为手，入库已 ×100）
    turnover    换手率（小数）
    amount      成交额（元）
    adjust      复权方式（""=不复权, qfq=前复权, hfq=后复权）
"""

import argparse
import contextlib
import datetime
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from download import akshare  # noqa: E402

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


def print_db_state(db_path, adjust):
    """打印主库 akshare_tx 每只股票最后日期分布。"""
    max_dates = akshare.get_max_dates(db_path, adjust)
    if not max_dates:
        print("[每日增量] 主库 akshare_tx 无记录，本次按首次全量路径抓取")
        return
    from collections import Counter
    dist = Counter(d.strftime("%Y-%m-%d") for d in max_dates.values())
    latest = max(max_dates.values())
    print(f"[每日增量] 主库已有 {len(max_dates)} 只股票行情，最新日期 {latest:%Y-%m-%d}")
    print("[每日增量] 各股最后日期分布（前 10 个日期）：")
    for d, c in sorted(dist.items(), reverse=True)[:10]:
        print(f"    {d}: {c} 只")


def run_daily(args):
    db_path = args.db or akshare.DEFAULT_DB_PATH
    started = datetime.datetime.now()
    print("=" * 60)
    print(f"[每日增量] 开始 {started:%Y-%m-%d %H:%M:%S}")
    print(f"[每日增量] 主库: {db_path}")
    print(f"[每日增量] 复权: {args.adjust or '不复权'}，排除 ST: {not args.include_st}，每股间隔 {args.delay}s")
    print("=" * 60)

    print_db_state(db_path, args.adjust)

    result = akshare.run(
        codes=None,               # stock_list 模式：从主库读在市股，逐股按库内最大日期算增量
        adjust=args.adjust,
        db_path=db_path,
        no_csv=True,
        exclude_st=not args.include_st,
        stage_path="auto",        # 暂存模式：先写暂存库，不碰主库
        delay=args.delay,
    )

    stage_path = result.get("stage_path")
    if stage_path and os.path.exists(stage_path):
        print("\n[每日增量] 抓取完成，开始合并暂存库入主库 ...")
        total, nfiles = akshare.merge_stages(stage_path, db_path=db_path)
        print(f"[每日增量] 合并完成：{nfiles} 个暂存文件，共 {total} 行 -> {db_path}")
    else:
        print("\n[每日增量] 无新数据，跳过合并（主库已是最新）")

    failed = result.get("failed") or {}
    elapsed = datetime.datetime.now() - started
    print("\n" + "=" * 60)
    print(f"[每日增量] 结束 {datetime.datetime.now():%Y-%m-%d %H:%M:%S}，耗时 {elapsed}")
    print(f"[每日增量] 本次入库 {result.get('rows_count', 0)} 行，失败 {len(failed)} 只")
    if failed:
        print(f"[每日增量] 失败清单: {result.get('failed_path')}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="每日 A 股行情增量更新（暂存 + 合并 + 日志）")
    parser.add_argument("--adjust", default="hfq", choices=["", "qfq", "hfq"],
                        help="复权方式：\"\"=不复权, qfq=前复权, hfq=后复权（默认）")
    parser.add_argument("--delay", type=float, default=1.0, help="每股间隔秒数，默认 1.0")
    parser.add_argument("--db", default=None, help="主库 DuckDB 路径（默认 download/autots.duckdb）")
    parser.add_argument("--include-st", action="store_true", help="不排除 ST 股（默认排除）")
    args = parser.parse_args()

    os.makedirs(TEMP_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(TEMP_DIR, f"daily_akshare_{stamp}.log")

    with open(log_path, "w", encoding="utf-8") as log_f:
        out_tee = _Tee(sys.stdout, log_f)
        err_tee = _Tee(sys.stderr, log_f)
        with contextlib.redirect_stdout(out_tee), contextlib.redirect_stderr(err_tee):
            try:
                run_daily(args)
                code = 0
            except Exception as e:
                print(f"\n[每日增量] 失败: {e}", file=sys.stderr)
                code = 1

    print(f"日志已保存: {log_path}")
    sys.exit(code)


if __name__ == "__main__":
    main()
