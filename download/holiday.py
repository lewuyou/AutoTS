# -*- coding: utf-8 -*-
"""节假日日历抓取入库模块（timor.tech，含周末、法定节假日、调休）。

可独立运行，也可由 download_all.py 调用 run()。

用法（独立运行）：
    python -m AutoTS.download.holiday                    # 抓当年全年
    python -m AutoTS.download.holiday --start 2013-01-01 --end 2026-12-31
    python -m AutoTS.download.holiday --years 2024,2025,2026

说明：
    * 接口 https://timor.tech/api/holiday/year/YYYY?type=Y&week=Y 返回全年每一天。
    * holiday=true  -> 放假（法定节假日 / 周末 / 调休后放假的周末）
    * holiday=false -> 调休上班日（周末上班）
    * 用 curl 带浏览器 UA 即可过 Cloudflare，不需要登录。

依赖：
    pip install duckdb
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

try:
    import duckdb
except ImportError:
    duckdb = None

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autots.duckdb")
TABLE_NAME = "holiday_calendar"


def _require_duckdb():
    if duckdb is None:
        raise ImportError("缺少 duckdb，请安装：python3 -m pip install duckdb")


def fetch_holiday_years(years):
    """用 curl 抓 timor.tech 节假日数据（含周末），返回 {year: {dateStr: info}}。"""
    out = {}
    for y in years:
        url = f"https://timor.tech/api/holiday/year/{y}?type=Y&week=Y"
        try:
            r = subprocess.run(
                [
                    "curl", "-s", "-f", url,
                    "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "-H", "Accept: application/json",
                ],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                print(f"  [节假日] {y} 抓取失败: curl exit {r.returncode}", file=sys.stderr)
                continue
            data = json.loads(r.stdout)
            if data.get("code") == 0 and data.get("holiday"):
                out[y] = data["holiday"]
        except Exception as e:
            print(f"  [节假日] {y} 抓取异常: {e}", file=sys.stderr)
    return out


def daily_dates(start, end):
    """逐日日期序列（含两端）。"""
    s = datetime.date.fromisoformat(start)
    e = datetime.date.fromisoformat(end)
    out = []
    d = s
    while d <= e:
        out.append(d.isoformat())
        d += datetime.timedelta(days=1)
    return out


def holiday_payload_to_rows(holiday_payload, start, end):
    """把 timor.tech 节假日数据展开成 (date, is_holiday, holiday_name, is_workday_adjustment, is_holiday_related) 行。

    接口带 ?type=Y&week=Y 后返回全年每一天：
      * holiday=true  -> 放假（法定节假日 / 周末 / 调休后放假的周末）
      * holiday=false -> 调休上班日（周末上班）
      * is_holiday_related = is_holiday OR is_workday_adjustment（节假日相关日期）
    """
    api_map = {}
    for year_data in holiday_payload.values():
        for date_str, info in year_data.items():
            d = info.get("date")
            if not d:
                continue
            api_map[d] = {
                "holiday": bool(info.get("holiday")),
                "name": info.get("name") or "",
            }

    rows = []
    for d in daily_dates(start, end):
        info = api_map.get(d)
        if info is None:
            # API 缺数据时保守标记为工作日
            rows.append((d, False, "", False, False))
            continue
        is_holiday = info["holiday"]
        name = info["name"]
        is_workday_adjustment = not is_holiday and datetime.date.fromisoformat(d).weekday() >= 5
        is_holiday_related = is_holiday or is_workday_adjustment
        rows.append((d, is_holiday, name, is_workday_adjustment, is_holiday_related))
    return rows


def ingest(rows, db_path=DEFAULT_DB_PATH):
    """UPSERT 到 DuckDB 的 holiday_calendar 表。"""
    _require_duckdb()
    con = duckdb.connect(db_path)
    try:
        # 检查表是否存在及列是否齐全
        table_exists = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
            (TABLE_NAME,)
        ).fetchone()
        
        if table_exists:
            cols = [c[1] for c in con.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()]
            if "is_holiday_related" not in cols:
                con.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN is_holiday_related BOOLEAN")
                # 历史数据补全
                con.execute(f"UPDATE {TABLE_NAME} SET is_holiday_related = is_holiday OR is_workday_adjustment")
        else:
            con.execute(
                f"""
                CREATE TABLE {TABLE_NAME} (
                    date DATE PRIMARY KEY,
                    is_holiday BOOLEAN,
                    holiday_name TEXT,
                    is_workday_adjustment BOOLEAN,
                    is_holiday_related BOOLEAN
                )
                """
            )
        
        con.executemany(
            f"INSERT OR REPLACE INTO {TABLE_NAME} (date, is_holiday, holiday_name, is_workday_adjustment, is_holiday_related) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
    finally:
        con.close()
    return len(rows)


def write_csv(rows, base_dir):
    """写节假日 CSV，返回文件路径。"""
    path = os.path.join(base_dir, "holiday_calendar.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date", "is_holiday", "holiday_name", "is_workday_adjustment", "is_holiday_related"])
        w.writerows(rows)
    return path


def run(start=None, end=None, years=None, db_path=None, no_csv=False, csv_dir=None):
    """供外部调用的入口。

    参数：
        start: 起始日期 YYYY-MM-DD，默认当年 1 月 1 日
        end: 结束日期 YYYY-MM-DD，默认当年 12 月 31 日
        years: 年份列表，如 [2024, 2025]，优先于 start/end
        db_path: DuckDB 文件路径，默认模块目录下 holiday.duckdb
        no_csv: 是否跳过 CSV 导出
        csv_dir: CSV 输出目录，默认模块目录
    返回：
        dict 包含 rows_count, db_path, csv_path, stats 等
    """
    today = datetime.date.today()

    if years:
        year_list = [int(y) for y in years]
        start_d = datetime.date(min(year_list), 1, 1)
        end_d = datetime.date(max(year_list), 12, 31)
    else:
        if start or end:
            start_d = datetime.date.fromisoformat(start) if start else datetime.date(today.year, 1, 1)
            end_d = datetime.date.fromisoformat(end) if end else datetime.date(today.year, 12, 31)
        else:
            start_d = datetime.date(today.year, 1, 1)
            end_d = datetime.date(today.year, 12, 31)
        year_list = list(range(start_d.year, end_d.year + 1))

    start_s = start_d.isoformat()
    end_s = end_d.isoformat()

    print(f"[节假日] 抓取年份：{year_list}，范围 {start_s} ~ {end_s}")
    payload = fetch_holiday_years(year_list)
    if not payload:
        raise RuntimeError("未抓取到任何节假日数据")

    rows = holiday_payload_to_rows(payload, start_s, end_s)
    n_holiday = sum(1 for r in rows if r[1])
    n_adj = sum(1 for r in rows if r[3])
    n_related = sum(1 for r in rows if r[4])
    print(f"[节假日] {n_holiday} 天放假（含周末），调休上班 {n_adj} 天，节假日相关共 {n_related} 天")

    db_path = db_path or DEFAULT_DB_PATH
    n = ingest(rows, db_path=db_path)
    print(f"[节假日] 入库 {n} 行 -> {db_path}")

    csv_path = None
    if not no_csv:
        csv_dir = csv_dir or os.path.dirname(os.path.abspath(__file__))
        csv_path = write_csv(rows, csv_dir)
        print(f"[节假日] CSV -> {csv_path}")

    return {
        "rows_count": n,
        "db_path": db_path,
        "csv_path": csv_path,
        "stats": {"total_days": len(rows), "holiday_days": n_holiday, "workday_adjustment_days": n_adj},
    }


def main():
    parser = argparse.ArgumentParser(description="节假日日历抓取入库（timor.tech）")
    parser.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD（默认当年 1 月 1 日）")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认当年 12 月 31 日）")
    parser.add_argument("--years", default=None, help="逗号分隔年份，如 2024,2025,2026（优先于 start/end）")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB 文件路径")
    parser.add_argument("--no-csv", action="store_true", help="不写 CSV，只入库")
    args = parser.parse_args()

    years = [int(y.strip()) for y in args.years.split(",") if y.strip()] if args.years else None
    run(start=args.start, end=args.end, years=years, db_path=args.db, no_csv=args.no_csv)


if __name__ == "__main__":
    main()
