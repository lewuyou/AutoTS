# -*- coding: utf-8 -*-
"""东方财富业绩报告抓取（接入模块，季频）。

仅保留数据源特有逻辑：东财 RPT_LICO_FN_CPD_BB 接口请求（curl + 本地代理）、
单页返回该股全部报告期（全量重抓 + UPSERT 幂等，每天重跑即可刷到最新披露）。
日志/建表/UPSERT/暂存路径/合并/结果统一走 download.common。

https://emdata.eastmoney.com/nbjb/detail.html?fc=300999&fn=%E9%87%91%E9%BE%99%E9%B1%BC

用法（独立运行）：
    python -m download.sources.nbjb                          # 从 stock_list 抓取全部 A 股业绩报告
    python -m download.sources.nbjb --codes 300999,601318    # 指定股票代码
    python -m download.sources.nbjb --stage auto             # 暂存模式
    python -m download.sources.nbjb --merge "download/temp/nbjb_stage_*.duckdb"

接口（无需 Cookie / token，GET 即可）：
    RPT_LICO_FN_CPD_BB: 业绩报告，按报告期倒序，单页可返回全部
    字段: BASIC_EPS 每股收益(元), TOTAL_OPERATE_INCOME 营业总收入(元) + _TQ 同比%,
          PARENT_NETPROFIT 归母净利润(元) + _TQ 同比%, REPORTDATE 报告期, NOTICE_DATE 公告日

入库表 nbjb 字段含义（季频，主键 (code, report_date)）：
    code                      股票代码（6 位数字）
    name                      股票简称
    report_date               报告期截止日（如 2021-03-31 表示 2021 年一季报期末）
    report_q                  报告期季度标识（如 "2021Q1"，来自 REPORTDATEWZ）
    report_label              报告期中文标签（如 "2021年 一季报"，来自 REPORTDATEYW）
    eps                       基本每股收益（元，来自 BASIC_EPS）
    total_operate_income      营业总收入（元，当季累计值）
    total_operate_income_yoy  营业总收入同比（%，来自 TOTAL_OPERATE_INCOME_TQ）
    parent_netprofit          归母净利润（元，当季累计值）
    parent_netprofit_yoy      归母净利润同比（%，来自 PARENT_NETPROFIT_TQ）
    notice_date               公告日期（实际披露日）
"""

import json
import subprocess
import time

from download.common import cli
from download.common import logging as common_logging
from download.common import results
from download.common import shared
from download.common import storage
from download.common.paths import DEFAULT_DB_PATH, TEMP_DIR

SOURCE_NAME = "nbjb"
TABLE_NAME = "nbjb"
TABLE_SCHEMA = """(
    code TEXT,
    name TEXT,
    report_date DATE,
    report_q TEXT,
    report_label TEXT,
    eps DOUBLE,
    total_operate_income DOUBLE,
    total_operate_income_yoy DOUBLE,
    parent_netprofit DOUBLE,
    parent_netprofit_yoy DOUBLE,
    notice_date DATE,
    PRIMARY KEY (code, report_date)
)"""
COLUMNS = [
    "code", "name", "report_date", "report_q", "report_label",
    "eps", "total_operate_income", "total_operate_income_yoy",
    "parent_netprofit", "parent_netprofit_yoy", "notice_date",
]

DEFAULT_DELAY = 0.3  # 每股抓取间隔（秒），避免触发东财限流
PAGE_SIZE = 200  # 单页返回该股全部报告期

API_BASE = "https://datacenter.eastmoney.com/securities/api/data/get"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
PROXY = "http://127.0.0.1:7897"

STY = ",".join([
    "SECURITY_CODE", "SECURITY_NAME_ABBR", "REPORTDATE", "REPORTDATEWZ", "REPORTDATEYW",
    "BASIC_EPS", "TOTAL_OPERATE_INCOME", "TOTAL_OPERATE_INCOME_TQ",
    "PARENT_NETPROFIT", "PARENT_NETPROFIT_TQ", "NOTICE_DATE",
])


def _to_date(s):
    """'2026-06-30 00:00:00' -> '2026-06-30'；空值/无日期返回 None（入库为 NULL）。"""
    d = (s or "").split(" ")[0]
    return d or None


def fetch_stock(code, end=None, log=None):
    """抓单只股票业绩报告（单页全量），返回行列表。

    end 给定时（统一截止日 "YYYY-MM-DD"），报告期晚于 end 的记录跳过不入库。
    """
    url = (
        f"{API_BASE}?type=RPT_LICO_FN_CPD_BB&source=DataCenter&client=WAP"
        f"&sty={STY}&p=1&ps={PAGE_SIZE}&sr=-1&st=REPORTDATE"
        f"&filter=(SECURITY_CODE=%22{code}%22)"
    )
    try:
        r = subprocess.run(
            ["curl", "-s", "-f", "-x", PROXY, url,
             "-H", f"User-Agent: {UA}", "-H", "Accept: application/json"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            log.error("%s curl exit %s", code, r.returncode)
            return []
        j = json.loads(r.stdout)
        data = (j.get("result") or {}).get("data") or []
    except Exception as e:
        log.error("%s 异常: %s", code, e)
        return []

    rows = []
    for rec in data:
        rd = _to_date(rec.get("REPORTDATE"))
        if not rd:
            continue  # 无报告期的记录跳过（主键必需）
        if end and rd > end:
            continue
        rows.append((
            code,
            rec.get("SECURITY_NAME_ABBR"),
            rd,
            rec.get("REPORTDATEWZ"),
            rec.get("REPORTDATEYW"),
            rec.get("BASIC_EPS"),
            rec.get("TOTAL_OPERATE_INCOME"),
            rec.get("TOTAL_OPERATE_INCOME_TQ"),
            rec.get("PARENT_NETPROFIT"),
            rec.get("PARENT_NETPROFIT_TQ"),
            _to_date(rec.get("NOTICE_DATE")),
        ))
    rows.sort(key=lambda r: (r[0], r[2]))
    return rows


merge_stages = storage.make_merge_stages(TABLE_NAME, TABLE_SCHEMA, COLUMNS, DEFAULT_DB_PATH)


def run(codes=None, end=None, db_path=None, delay=DEFAULT_DELAY, exclude_st=True,
        stage_path=None, run_id=""):
    """执行业绩报告抓取，写入暂存库（或主库），不在此合并。

    参数：
        codes: 股票代码列表；为 None 时从 stock_list 表读在市股票（排除 ST）
        end: 统一截止日期 YYYY-MM-DD；报告期晚于该日的记录跳过不入库（None 则全量）
        db_path: 主库路径，默认 download/autots.duckdb
        delay: 每股抓取间隔秒数（限速防封）
        exclude_st: stock_list 模式排除名称含 ST 的股票（默认 True）
        stage_path: 暂存 DuckDB 路径；"auto" 自动生成，None 直入主库
    返回：
        RunResult（fetch_status/rows_staged/failed 等）
    """
    log = common_logging.get_logger(SOURCE_NAME, run_id)
    db_path = db_path or DEFAULT_DB_PATH
    res = results.RunResult(SOURCE_NAME, run_id)

    if end:
        end = str(end).strip().replace("/", "-")

    stage_path = storage.make_stage_path(SOURCE_NAME, TEMP_DIR, stage_path)
    if stage_path:
        log.info("暂存模式：写入 %s，事后用 merge 合并入主库", stage_path)
    res.stage_path = stage_path
    ingest_db = stage_path or db_path

    if isinstance(codes, str):
        codes = [c.strip() for c in codes.split(",") if c.strip()]

    if codes is None:
        entries_all = shared.load_stock_list(db_path)
        n_st = sum(1 for e in entries_all if shared.is_st_name(e[2]))
        codes = [s for s, _d, n in entries_all if not exclude_st or not shared.is_st_name(n)]
        log.info("从 stock_list 读取 %d 只，排除 ST %d 只，待抓 %d 只", len(entries_all), n_st, len(codes))
    elif not codes:
        raise ValueError("未提供股票代码")

    total_rows = 0
    series = {}
    failed = {}

    for idx, code in enumerate(codes, 1):
        try:
            rows = fetch_stock(code, end=end, log=log)
        except Exception as e:
            failed[code] = str(e)
            log.error("%s 失败: %s", code, e)
            time.sleep(delay)
            continue
        if not rows:
            failed[code] = "无数据"
            log.warning("%s 无数据", code)
            time.sleep(delay)
            continue
        n = storage.ingest(ingest_db, TABLE_NAME, TABLE_SCHEMA, COLUMNS, rows)
        total_rows += n
        series[code] = {"n": n, "first": rows[0][2], "last": rows[-1][2]}
        if idx % 100 == 0 or idx == len(codes):
            log.info("[%d/%d] 进度：成功 %d 失败 %d 累计入库 %d 行",
                     idx, len(codes), len(series), len(failed), total_rows)
        else:
            log.info("[%d/%d] %s 入库 %d 行 %s ~ %s", idx, len(codes), code, n, rows[0][2], rows[-1][2])
        time.sleep(delay)

    if failed:
        res.failed_path = results.failed_path_for(SOURCE_NAME, TEMP_DIR)
        results.write_failed_list(failed, res.failed_path)
        log.warning("%d 只失败，清单: %s", len(failed), res.failed_path)

    log.info("完成：成功 %d 只，失败 %d 只，累计入库 %d 行", len(series), len(failed), total_rows)

    res.rows_staged = total_rows
    res.success = len(series)
    res.failed = len(failed)
    res.detail = {"series": series}
    if failed:
        if not series:
            res.fetch_status = results.FETCH_FAILED
            res.error = f"全部失败: {failed}"
        else:
            res.fetch_status = results.FETCH_PARTIAL
    else:
        res.fetch_status = results.FETCH_OK
    return res.finish()


def _add_args(parser):
    parser.add_argument("--codes", default=None, help="逗号分隔股票代码，如 300999,601318；不传则从 stock_list 表读取全部")
    parser.add_argument("--end", default=None, help="统一截止日期 YYYY-MM-DD，报告期晚于该日的记录不入库（不传则全量）")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"每股间隔秒数，默认 {DEFAULT_DELAY}")
    parser.add_argument("--include-st", action="store_true", help="stock_list 模式下不排除 ST 股票")
    parser.add_argument("--stage", default=None, help="暂存 DuckDB 路径；auto 自动生成，不设则直入主库")


def _build_kwargs(args, db_path):
    return dict(
        codes=args.codes,
        end=args.end,
        db_path=db_path,
        delay=args.delay,
        exclude_st=not args.include_st,
        stage_path=args.stage,
    )


def _log_start(log, args, db_path):
    log.info("开始，主库: %s，排除 ST: %s，每股间隔 %ss",
             db_path, not args.include_st, args.delay)


def main():
    cli.standard_main(
        source_name=SOURCE_NAME,
        description="东方财富业绩报告抓取入库（EPS/营收/净利润，季频）",
        add_args=_add_args,
        kwargs_builder=_build_kwargs,
        run_func=run,
        merge_func=merge_stages,
        default_db_path=DEFAULT_DB_PATH,
        temp_dir=TEMP_DIR,
        log_start=_log_start,
    )


if __name__ == "__main__":
    main()
