# -*- coding: utf-8 -*-
"""timor.tech 节假日日历抓取（接入模块，日频，含周末、法定节假日、调休）。

仅保留数据源特有逻辑：timor.tech 年度接口请求（curl）、payload 展开为逐日行。
日志/建表/UPSERT/暂存路径/合并/逐日序列/结果统一走 download.common。
整年 UPSERT 可重跑（节假日安排可能事后调整，不调用水位增量，每年全量覆盖）。

用法（独立运行）：
    python -m download.sources.holiday                    # 抓当年全年
    python -m download.sources.holiday --start 2013-01-01 --end 2026-12-31
    python -m download.sources.holiday --years 2024,2025,2026
    python -m download.sources.holiday --stage auto       # 暂存模式
    python -m download.sources.holiday --merge "download/temp/holiday_stage_*.duckdb"

接口（用 curl 带浏览器 UA 即可过 Cloudflare，不需要登录）：
    https://timor.tech/api/holiday/year/YYYY?type=Y&week=Y 返回全年每一天
    holiday=true  -> 放假（法定节假日 / 周末 / 调休后放假的周末）
    holiday=false -> 调休上班日（周末上班）

入库表 holiday_calendar 字段含义：
    date                  日期（日频，主键）
    is_holiday            是否放假（法定节假日 / 周末 / 调休后放假的周末）
    holiday_name          节假日名称（如 "国庆节"，普通周末/工作日为空字符串）
    is_workday_adjustment 是否调休上班日（周末上班的调休日）
    is_holiday_related    是否节假日相关日期 = is_holiday OR is_workday_adjustment
"""

import datetime
import json
import subprocess

from download.common import cli
from download.common import dates
from download.common import logging as common_logging
from download.common import results
from download.common import storage
from download.common.paths import DEFAULT_DB_PATH, TEMP_DIR

SOURCE_NAME = "holiday"
TABLE_NAME = "holiday_calendar"
TABLE_SCHEMA = """(
    date DATE,
    is_holiday BOOLEAN,
    holiday_name TEXT,
    is_workday_adjustment BOOLEAN,
    is_holiday_related BOOLEAN,
    PRIMARY KEY (date)
)"""
COLUMNS = ["date", "is_holiday", "holiday_name", "is_workday_adjustment", "is_holiday_related"]

API_BASE = "https://timor.tech/api/holiday/year"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def fetch_year(year, log):
    """抓 timor.tech 单年节假日数据（含周末），返回 {dateStr: info}；失败返回 {}。"""
    url = f"{API_BASE}/{year}?type=Y&week=Y"
    try:
        r = subprocess.run(
            ["curl", "-s", "-f", url,
             "-H", f"User-Agent: {UA}", "-H", "Accept: application/json"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            log.error("%d 抓取失败: curl exit %s", year, r.returncode)
            return {}
        data = json.loads(r.stdout)
        if data.get("code") == 0 and data.get("holiday"):
            return data["holiday"]
        log.error("%d 返回异常: code=%s", year, data.get("code"))
        return {}
    except Exception as e:
        log.error("%d 抓取异常: %s", year, e)
        return {}


def payload_to_rows(payload, start, end):
    """把 {year: {dateStr: info}} 展开成 COLUMNS 顺序的逐日行（仅覆盖抓取成功的年份）。

    接口带 ?type=Y&week=Y 后返回全年每一天；个别缺日保守标记为工作日，
    整年缺失（抓取失败）直接跳过，不写错误数据污染日历。
    """
    fetched_years = {int(y) for y in payload}
    api_map = {}
    for year_data in payload.values():
        for info in year_data.values():
            d = info.get("date")
            if d:
                api_map[d] = (bool(info.get("holiday")), info.get("name") or "")
    rows = []
    for d in dates.daily_dates(start, end):
        if int(d[:4]) not in fetched_years:
            continue
        is_holiday, name = api_map.get(d, (False, ""))
        is_adj = not is_holiday and datetime.date.fromisoformat(d).weekday() >= 5
        rows.append((d, is_holiday, name, is_adj, is_holiday or is_adj))
    return rows


merge_stages = storage.make_merge_stages(TABLE_NAME, TABLE_SCHEMA, COLUMNS, DEFAULT_DB_PATH)


def run(years=None, start=None, end=None, db_path=None, stage_path=None, run_id=""):
    """执行节假日日历抓取，写入暂存库（或主库），不在此合并。

    参数：
        years: 年份列表或逗号分隔字符串，如 [2024, 2025]；优先于 start/end
        start: 起始日期 YYYY-MM-DD，默认当年 1 月 1 日
        end: 截止日期 YYYY-MM-DD，默认当年 12 月 31 日
        db_path: 主库路径，默认 download/autots.duckdb
        stage_path: 暂存 DuckDB 路径；"auto" 自动生成，None 直入主库
    返回：
        RunResult（fetch_status/rows_staged/failed 等）
    """
    log = common_logging.get_logger(SOURCE_NAME, run_id)
    db_path = db_path or DEFAULT_DB_PATH
    res = results.RunResult(SOURCE_NAME, run_id)

    today = datetime.date.today()
    if years:
        if isinstance(years, str):
            years = [int(y.strip()) for y in years.split(",") if y.strip()]
        start_d = datetime.date(min(years), 1, 1)
        end_d = datetime.date(max(years), 12, 31)
    else:
        start_d = datetime.date.fromisoformat(start) if start else datetime.date(today.year, 1, 1)
        end_d = datetime.date.fromisoformat(end) if end else datetime.date(today.year, 12, 31)
    if end_d < start_d:
        raise ValueError(f"截止日期 {end_d} 早于起始日期 {start_d}")
    year_list = list(range(start_d.year, end_d.year + 1))

    stage_path = storage.make_stage_path(SOURCE_NAME, TEMP_DIR, stage_path)
    if stage_path:
        log.info("暂存模式：写入 %s，事后用 merge 合并入主库", stage_path)
    res.stage_path = stage_path
    ingest_db = stage_path or db_path

    log.info("抓取年份 %s，范围 %s ~ %s", year_list, start_d, end_d)
    payload = {}
    for y in year_list:
        data = fetch_year(y, log)
        if data:
            payload[y] = data
            log.info("%d 抓取成功，共 %d 天", y, len(data))

    res.success = len(payload)
    res.failed = len(year_list) - len(payload)
    if not payload:
        res.fetch_status = results.FETCH_FAILED
        res.error = "全部年份抓取失败"
        return res.finish()

    rows = payload_to_rows(payload, start_d.isoformat(), end_d.isoformat())
    n = storage.ingest(ingest_db, TABLE_NAME, TABLE_SCHEMA, COLUMNS, rows)
    res.rows_staged = n

    n_holiday = sum(1 for r in rows if r[1])
    n_adj = sum(1 for r in rows if r[3])
    log.info("入库 %d 行：放假 %d 天（含周末），调休上班 %d 天，节假日相关共 %d 天",
             n, n_holiday, n_adj, n_holiday + n_adj)
    res.detail = {"years": sorted(payload), "holiday_days": n_holiday, "workday_adjustment_days": n_adj}
    res.fetch_status = results.FETCH_PARTIAL if res.failed else results.FETCH_OK
    return res.finish()


def _add_args(parser):
    parser.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD（默认当年 1 月 1 日）")
    parser.add_argument("--end", default=None, help="截止日期 YYYY-MM-DD（默认当年 12 月 31 日）")
    parser.add_argument("--years", default=None, help="逗号分隔年份，如 2024,2025,2026（优先于 start/end）")
    parser.add_argument("--stage", default=None, help="暂存 DuckDB 路径；auto 自动生成，不设则直入主库")


def _build_kwargs(args, db_path):
    return dict(
        years=args.years,
        start=args.start,
        end=args.end,
        db_path=db_path,
        stage_path=args.stage,
    )


def _log_start(log, args, db_path):
    log.info("开始，主库: %s，年份: %s，范围: %s ~ %s",
             db_path, args.years or "当年", args.start or "当年 1 月 1 日", args.end or "当年 12 月 31 日")


def main():
    cli.standard_main(
        source_name=SOURCE_NAME,
        description="timor.tech 节假日日历抓取入库（日频，含周末与调休）",
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
