# -*- coding: utf-8 -*-
"""统一执行流程：日志、单源隔离、暂存合并、汇总与退出码。

执行分三段：holiday 先行串行（抓取+合并，其他源依赖主库交易日历判断已齐全交易日）
→ 其余源并行抓取（各自只写暂存，线程间互不干扰）→ 按序串行合并主库。
任一数据源异常只影响自身，不阻断其他数据源；最后汇总并决定退出码。
"""

import datetime
import glob
import os

from concurrent.futures import ThreadPoolExecutor, as_completed

from download import registry
from download.common import logging as common_logging
from download.common import results
from download.common.paths import TEMP_DIR


def resolve_sources(spec, include_baidu=False):
    """把 --sources 参数解析成数据源名列表。

    spec 为 daily（默认）时按 DAILY_ORDER 展开，但 baidu 需 include_baidu=True 才包含
    （百度指数限速严格、耗时最长，默认不每日跑）；显式列出的源（含 baidu）不受此限制。
    """
    if spec in (None, "", "daily"):
        names = list(registry.DAILY_ORDER)
        if not include_baidu:
            names = [n for n in names if n != "baidu"]
        return names
    if spec == "all":
        return list(registry.SOURCES.keys())
    names = [n.strip() for n in spec.split(",") if n.strip()]
    for n in names:
        if n not in registry.SOURCES:
            raise ValueError(f"未知数据源: {n}（可选: {', '.join(registry.SOURCES)}）")
    return names


def _merge_if_needed(mod, res, db_path, log):
    """抓取完成后把全部暂存（本次 + 遗留未合并）并入主库；无暂存则跳过。

    合并失败保留暂存并标记 pending。
    """
    # 与增量水位 pending_stage_files 认领口径一致：水位已认领的遗留暂存也必须在此并入主库
    pattern = os.path.join(TEMP_DIR, f"{res.source}_stage_*.duckdb")
    pending = [f for f in glob.glob(pattern) if not f.endswith(".merged")]
    if not pending:
        res.merge_status = results.MERGE_SKIPPED
        return
    try:
        total, nfiles = mod.merge_stages(pattern, db_path=db_path)
        res.rows_merged = total
        res.merge_status = results.MERGE_MERGED
        log.info("合并完成：%d 个文件，共 %d 行 -> %s", nfiles, total, db_path)
    except Exception as e:
        res.merge_status = results.MERGE_PENDING
        suffix = f"合并失败: {e}"
        res.error = f"{res.error}; {suffix}" if res.error else suffix
        log.error("合并失败（暂存保留，可重跑合并）: %s", e)


def _merge_leftovers(mod, name, db_path, log):
    """抓取开始前，把遗留未合并暂存并入主库；合并失败只告警不阻断。

    上次运行在抓取阶段被中断时，其暂存会遗留；若等本次抓取完成才合并，
    长任务再次中断会导致数据一直停在暂存。故抓取前先清理遗留。
    """
    pattern = os.path.join(TEMP_DIR, f"{name}_stage_*.duckdb")
    leftovers = [f for f in glob.glob(pattern) if not f.endswith(".merged")]
    if not leftovers:
        return
    try:
        total, nfiles = mod.merge_stages(pattern, db_path=db_path)
        log.info("合并遗留暂存 %d 个文件，共 %d 行 -> %s", nfiles, total, db_path)
    except Exception as e:
        log.error("合并遗留暂存失败（继续抓取，可稍后手动 merge）: %s", e)


def _cleanup_merged_stages(log):
    """运行前清理已合并暂存文件（.merged 后缀）：数据已入主库，文件只剩占空间。"""
    removed = 0
    for f in glob.glob(os.path.join(TEMP_DIR, "*_stage_*.duckdb.merged")):
        try:
            os.remove(f)
            removed += 1
        except OSError as e:
            log.error("清理已合并暂存失败 %s: %s", f, e)
    if removed:
        log.info("清理已合并暂存文件 %d 个", removed)


def _fetch_one(name, mod, kwargs, run_id):
    """单个数据源的抓取阶段（线程内执行）：只写暂存，不碰主库；异常收敛到 RunResult。"""
    log = common_logging.get_logger(name, run_id)
    log.info("开始抓取数据源 %s", name)
    res = results.RunResult(name, run_id)
    try:
        res = mod.run(**kwargs)
    except Exception as e:
        log.exception("数据源 %s 执行失败: %s", name, e)
        res.fetch_status = results.FETCH_FAILED
        res.error = str(e)
    res.finish()
    return res


def run(args):
    """执行指定数据源的 daily 抓取（暂存 + 合并），返回退出码。

    流程：遗留暂存合并（串行）→ holiday 抓取+合并（串行先行）
    → 其余源并行抓取 → 全部源按序串行合并主库。
    """
    names = resolve_sources(args.sources, include_baidu=getattr(args, "with_baidu", False))
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(TEMP_DIR, exist_ok=True)
    log_path = os.path.join(TEMP_DIR, f"daily_{run_id}.log")
    common_logging.setup_logging(log_path, run_id)

    root = common_logging.get_logger("daily", run_id)
    root.info("=" * 60)
    root.info("每日更新开始，共 %d 个数据源: %s", len(names), ", ".join(names))
    root.info("日志: %s", log_path)
    root.info("=" * 60)

    _cleanup_merged_stages(root)

    # 预加载模块、构建参数、解析主库路径
    prepared = {}
    for name in names:
        entry = registry.SOURCES[name]
        mod = registry.load_source(name)
        kwargs = dict(entry["daily_kwargs"](args), run_id=run_id)
        db_path = kwargs.get("db_path") or mod.DEFAULT_DB_PATH
        prepared[name] = (mod, kwargs, db_path)

    # 抓取前先串行清理遗留暂存（写主库，避免与并行抓取交错）
    if not args.stage_only:
        for name in names:
            mod, _, db_path = prepared[name]
            _merge_leftovers(mod, name, db_path, common_logging.get_logger(name, run_id))

    # holiday 先行：其他源抓的时候要读主库 holiday_calendar 判断已齐全交易日
    fetched = {}
    rest = list(names)
    if "holiday" in rest:
        rest.remove("holiday")
        mod, kwargs, db_path = prepared["holiday"]
        res = _fetch_one("holiday", mod, kwargs, run_id)
        if not args.stage_only:
            _merge_if_needed(mod, res, db_path, common_logging.get_logger("holiday", run_id))
        fetched["holiday"] = res

    # 其余源并行抓取（各自只写暂存文件，单源内仍单线程）
    if rest:
        root.info("并行抓取 %d 个数据源: %s", len(rest), ", ".join(rest))
        with ThreadPoolExecutor(max_workers=len(rest)) as ex:
            futures = {ex.submit(_fetch_one, n, prepared[n][0], prepared[n][1], run_id): n
                       for n in rest}
            for fut in as_completed(futures):
                res = fut.result()
                fetched[res.source] = res
                common_logging.get_logger(res.source, run_id).info(
                    "数据源 %s 抓取结束：抓取 %s，暂存 %d 行，失败 %d",
                    res.source, res.fetch_status, res.rows_staged, res.failed)

    # 合并主库：全部抓取完成后按序串行收尾，避免多线程并发写主库
    collected = []
    for name in names:
        res = fetched[name]
        mod, _, db_path = prepared[name]
        log = common_logging.get_logger(name, run_id)
        if not args.stage_only and name != "holiday":
            _merge_if_needed(mod, res, db_path, log)
        collected.append(res)
        log.info("数据源 %s 结束：抓取 %s / 合并 %s，暂存 %d 行，合并 %d 行，失败 %d",
                 name, res.fetch_status, res.merge_status,
                 res.rows_staged, res.rows_merged, res.failed)

    return _summarize(collected, run_id, log_path)


def _summarize(collected, run_id, log_path):
    """打印汇总，返回退出码（0=全部正常，1=存在失败或待合并，2=参数错误由调用方处理）。"""
    root = common_logging.get_logger("daily", run_id)
    root.info("")
    root.info("=" * 60)
    root.info("汇总")
    root.info("=" * 60)
    code = 0
    for res in collected:
        if not res.ok:
            code = 1
        status = "成功" if res.ok else "失败"
        root.info("  %s: %s（抓取 %s / 合并 %s，暂存 %d 行，合并 %d 行，失败 %d）",
                  res.source, status, res.fetch_status, res.merge_status,
                  res.rows_staged, res.rows_merged, res.failed)
        if res.error:
            root.error("    %s", res.error)
        if res.failed_path:
            root.info("    失败清单: %s", res.failed_path)
    root.info("日志已保存: %s", log_path)
    return code
