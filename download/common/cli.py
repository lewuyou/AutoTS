# -*- coding: utf-8 -*-
"""数据源独立 CLI 的统一脚手架。

各 sources 模块的 main() 只声明自身参数、组装 run 入参和开始日志，
--db/--merge 分支、run_id 生成、日志初始化、结束日志与退出码统一走这里，
保证新增数据源时 CLI 行为一致。
"""

import argparse
import datetime
import os
import sys

from download.common import logging as common_logging


def standard_main(source_name, description, add_args, kwargs_builder, run_func,
                  merge_func, default_db_path, temp_dir, log_start):
    """标准数据源 CLI 入口。

    参数：
        add_args(parser)        追加该数据源自己的 argparse 参数（不含 --db/--merge）
        kwargs_builder(args, db_path)  由解析结果构造 run(**kw) 入参（不含 run_id）
        run_func(**kw)          执行抓取，返回 RunResult
        merge_func(pattern, db_path=...)  暂存合并函数
        log_start(log, args, db_path)  打印"开始"上下文日志
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--db", default=None, help="主库 DuckDB 路径（默认 download/autots.duckdb）")
    parser.add_argument("--merge", default=None, help="合并模式：暂存文件 glob，如 \"download/temp/*_stage_*.duckdb\"")
    add_args(parser)
    args = parser.parse_args()

    db_path = args.db or default_db_path
    if args.merge:
        total, nfiles = merge_func(args.merge, db_path=db_path)
        print(f"合并完成：{nfiles} 个文件，共 {total} 行 -> {db_path}")
        return

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(temp_dir, exist_ok=True)
    log_path = os.path.join(temp_dir, f"{source_name}_{run_id}.log")
    common_logging.setup_logging(log_path, run_id)
    log = common_logging.get_logger(source_name, run_id)
    log_start(log, args, db_path)

    res = run_func(**kwargs_builder(args, db_path), run_id=run_id)

    log.info("结束，耗时 %s，状态 %s / %s", res.elapsed, res.fetch_status, res.merge_status)
    log.info("日志已保存: %s", log_path)
    sys.exit(0 if res.ok else 1)
