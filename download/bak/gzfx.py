# -*- coding: utf-8 -*-
"""东方财富估值通道抓取入库模块（市盈率/市净率/市销率/市现率，日频）。
https://emdata.eastmoney.com/gzfx/detail.html?fc=300999.SZ&fn=%E9%87%91%E9%BE%99%E9%B1%BC
可独立运行，也可由 download_all.py 调用 run()。

用法（独立运行）：
    python -m download.gzfx                          # 从 stock_list 增量抓取全部 A 股（近1年日频）
    python -m download.gzfx --codes 300999,601318    # 指定股票代码
    python -m download.gzfx --datetype 4             # 指定口径（4=近10年月频，仅手动回补用）
    python -m download.gzfx --stage auto             # 暂存模式，事后 --merge 合并入主库

模式说明：
    * 不传 --codes 时，从 DuckDB stock_list 表读取全部在市股票（自动排除名称含 ST 的）。
    * 本模块只做每日增量：默认 date_type=1（近1年日频），UPSERT 幂等，每天重跑即可刷到最新。
      历史全量数据由其他渠道入库，不经此脚本。
    * 失败清单写入 download/temp/gzfx_failed_<时间戳>.txt，重跑即可补抓。

暂存模式（--stage）：
    数据不入主库，改写入暂存 DuckDB（表结构与主库 gzfx 完全相同），
    供多个并行任务各自生成暂存文件后，用 --merge 统一合并入主库：
        python -m download.gzfx --stage auto --codes 000001,600000
        python -m download.gzfx --merge "download/temp/gzfx_stage_*.duckdb"
    合并成功的暂存文件自动重命名加 .merged 后缀，防止重复合并。

接口（无需 Cookie / token，GET 即可）：
    估值走势 RPT_CUSTOM_DMSK_TREND: INDICATOR_VALUE 实际估值
    估值通道 RPT_CUSTOM_DMSK:       STOCK_PRICE 股价 + PASS1~5 通道带价格 + MULTIPLE1~5 倍数
    INDICATORTYPE: 1=市盈率 2=市净率 3=市销率 4=市现率（必填，不可省略）
    DATETYPE:      1=近1年(日频) 2=近3年(周频) 3=近5年(周频) 4=近10年(月频)

入库表 gzfx 字段含义（按 (code, indicator, date) 合并走势与通道两类数据）：
    code         股票代码（6 位数字）
    indicator    指标类型：pe=市盈率 pb=市净率 ps=市销率 pcf=市现率
    date         交易日期
    value        实际估值（PE/PB/PS/PCF 的 TTM 值，来自走势接口 INDICATOR_VALUE）
    stock_price  股价（元，来自通道接口 STOCK_PRICE）
    pass1~pass5  估值通道 5 条带的下轨~上轨价格（元），即股价处于该带时对应估值为 mult1~mult5
    mult1~mult5  通道带对应的估值倍数（如 pe 的 5 档倍数），pass1~5 = 每股净资产/盈利等 × mult1~5
    注：走势数据日频/周频每天都有 value；通道数据仅在通道带重算日有 pass/mult，其余日为 None。

依赖：
    pip install duckdb
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

try:
    import duckdb
except ImportError:
    duckdb = None

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autots.duckdb")
TABLE_NAME = "gzfx"

INDICATOR_TYPES = {1: "pe", 2: "pb", 3: "ps", 4: "pcf"}  # 市盈/市净/市销/市现
DATE_TYPES = {1: "1y", 2: "3y", 3: "5y", 4: "10y"}        # 1=日频 2/3=周频 4=月频
DEFAULT_DELAY = 0.3  # 每股抓取间隔（秒），避免触发东财限流

API_BASE = "https://datacenter.eastmoney.com/securities/api/data/get"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def _require_duckdb():
    if duckdb is None:
        raise ImportError("缺少 duckdb，请安装：python3 -m pip install duckdb")


def fetch_api(report_type, code, indicator_type, date_type):
    """GET datacenter 接口，返回 data 列表。失败返回 []。"""
    filt = f"(SECURITY_CODE%3D%22{code}%22)(INDICATORTYPE%3D{indicator_type})(DATETYPE%3D{date_type})"
    url = (
        f"{API_BASE}?type={report_type}&p=1&sr=-1&st=TRADE_DATE"
        f"&var=source=DataCenter&client=WAP&filter={filt}"
    )
    try:
        r = subprocess.run(
            ["curl", "-s", "-f", "-x", "http://127.0.0.1:7897", url,
             "-H", f"User-Agent: {UA}", "-H", "Accept: application/json"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            print(f"  [估值] {code} type={report_type} it={indicator_type} curl exit {r.returncode}", file=sys.stderr)
            return []
        j = json.loads(r.stdout)
        return (j.get("result") or {}).get("data") or []
    except Exception as e:
        print(f"  [估值] {code} type={report_type} it={indicator_type} 异常: {e}", file=sys.stderr)
        return []


def _to_date(s):
    """'2026-09-18 00:00:00' -> '2026-09-18'"""
    return (s or "").split(" ")[0]


def fetch_stock(code, date_type=1):
    """抓单只股票 4 个指标的走势 + 通道，返回行列表。

    行格式: (code, indicator, date, value, stock_price, pass1..pass5, mult1..mult5)
    走势行 value=INDICATOR_VALUE，通道字段为 None；通道行反之。
    这里把两类按 (code, indicator, date) 合并成一行。
    """
    merged = {}  # (indicator, date) -> dict
    for it, iname in INDICATOR_TYPES.items():
        for rec in fetch_api("RPT_CUSTOM_DMSK_TREND", code, it, date_type):
            d = _to_date(rec.get("TRADE_DATE"))
            if not d:
                continue
            m = merged.setdefault((iname, d), {})
            m["value"] = rec.get("INDICATOR_VALUE")
        for rec in fetch_api("RPT_CUSTOM_DMSK", code, it, date_type):
            d = _to_date(rec.get("TRADE_DATE"))
            if not d:
                continue
            m = merged.setdefault((iname, d), {})
            m["stock_price"] = rec.get("STOCK_PRICE")
            for i in range(1, 6):
                m[f"pass{i}"] = rec.get(f"PASS{i}")
                m[f"mult{i}"] = rec.get(f"MULTIPLE{i}")

    rows = []
    for (iname, d), m in merged.items():
        rows.append((
            code, iname, d,
            m.get("value"), m.get("stock_price"),
            m.get("pass1"), m.get("pass2"), m.get("pass3"), m.get("pass4"), m.get("pass5"),
            m.get("mult1"), m.get("mult2"), m.get("mult3"), m.get("mult4"), m.get("mult5"),
        ))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return rows


def _create_table(con):
    """建 gzfx 表（幂等）。"""
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            code TEXT,
            indicator TEXT,
            date DATE,
            value DOUBLE,
            stock_price DOUBLE,
            pass1 DOUBLE, pass2 DOUBLE, pass3 DOUBLE, pass4 DOUBLE, pass5 DOUBLE,
            mult1 DOUBLE, mult2 DOUBLE, mult3 DOUBLE, mult4 DOUBLE, mult5 DOUBLE,
            PRIMARY KEY (code, indicator, date)
        )
        """
    )


def ingest(rows, db_path=DEFAULT_DB_PATH):
    """UPSERT 到 DuckDB 的 gzfx 表。"""
    _require_duckdb()
    con = duckdb.connect(db_path)
    try:
        _create_table(con)
        con.executemany(
            f"""INSERT OR REPLACE INTO {TABLE_NAME}
                (code, indicator, date, value, stock_price,
                 pass1, pass2, pass3, pass4, pass5,
                 mult1, mult2, mult3, mult4, mult5)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    finally:
        con.close()
    return len(rows)


def load_stock_list(db_path=DEFAULT_DB_PATH, exclude_st=True):
    """从 stock_list 表读取 (symbol, name)。exclude_st=True 时排除名称含 ST 的股票。"""
    _require_duckdb()
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            "SELECT symbol, COALESCE(name, '') FROM stock_list ORDER BY symbol"
        ).fetchall()
    finally:
        con.close()
    entries = [(str(s), n or "") for s, n in rows]
    if exclude_st:
        entries = [e for e in entries if "ST" not in e[1].upper()]
    return entries


def write_failed(failed):
    """失败清单写文件，返回路径；无失败返回 None。"""
    if not failed:
        return None
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(temp_dir, f"gzfx_failed_{stamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("code\terror\n")
        for code in sorted(failed):
            f.write(f"{code}\t{failed[code]}\n")
    return path


def merge_stages(pattern, db_path=DEFAULT_DB_PATH):
    """把暂存 DuckDB 文件统一合并入主库 gzfx 表；已合并文件加 .merged 后缀。

    返回 (合并总行数, 合并文件数)。
    """
    import glob
    files = sorted(f for f in glob.glob(pattern) if not f.endswith(".merged"))
    if not files:
        raise ValueError(f"未找到暂存文件: {pattern}")
    _require_duckdb()
    con = duckdb.connect(db_path)
    total = 0
    try:
        _create_table(con)
        cols = ("code, indicator, date, value, stock_price, "
                "pass1, pass2, pass3, pass4, pass5, "
                "mult1, mult2, mult3, mult4, mult5")
        for f in files:
            con.execute(f"ATTACH '{f}' AS stage (READ_ONLY)")
            stage_tables = {r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_catalog = 'stage'"
            ).fetchall()}
            if TABLE_NAME not in stage_tables:
                con.execute("DETACH stage")
                os.rename(f, f + ".merged")
                print(f"[估值] 合并 {f}: 0 行（空暂存）")
                continue
            n = con.execute(f"SELECT COUNT(*) FROM stage.{TABLE_NAME}").fetchone()[0]
            con.execute(
                f"INSERT OR REPLACE INTO {TABLE_NAME} ({cols}) "
                f"SELECT {cols} FROM stage.{TABLE_NAME}"
            )
            con.execute("DETACH stage")
            os.rename(f, f + ".merged")
            total += n
            print(f"[估值] 合并 {f}: {n} 行")
    finally:
        con.close()
    print(f"[估值] 合并完成：{len(files)} 个文件，共 {total} 行 -> {db_path}")
    return total, len(files)


def run(codes=None, date_type=1, db_path=None, stage_path=None,
        exclude_st=True, delay=DEFAULT_DELAY):
    """供外部调用的入口。

    参数：
        codes: 股票代码列表；为 None 时从 stock_list 表读取在市股票（stock_list 模式）
        date_type: 抓取口径，默认 1=近1年(日频，每日增量)；2=近3年 3=近5年 4=近10年（手动回补用）
        db_path: 主库 DuckDB 文件路径
        stage_path: 暂存 DuckDB 路径；"auto" 自动生成 download/temp/gzfx_stage_<时间戳>_<pid>.duckdb；
                    设置后数据写入暂存库而非主库，事后用 merge_stages 统一合并
        exclude_st: stock_list 模式下排除名称含 ST 的股票（默认 True）
        delay: 每股抓取间隔秒数（限速防封）
    返回：
        dict 包含 rows_count, db_path, series, failed_path, stage_path 等
    """
    db_path = db_path or DEFAULT_DB_PATH

    if stage_path == "auto":
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stage_path = os.path.join(temp_dir, f"gzfx_stage_{ts}_{os.getpid()}.duckdb")
    if stage_path:
        print(f"[估值] 暂存模式：写入 {stage_path}，事后用 --merge 合并入主库")
    ingest_db = stage_path or db_path

    if isinstance(codes, str):
        codes = [c.strip() for c in codes.split(",") if c.strip()]

    stock_list_mode = codes is None
    if stock_list_mode:
        entries_all = load_stock_list(db_path, exclude_st=False)
        n_st = sum(1 for e in entries_all if "ST" in e[1].upper())
        codes = [s for s, n in entries_all if not exclude_st or "ST" not in n.upper()]
        print(f"[估值] 从 stock_list 读取 {len(entries_all)} 只，排除 ST {n_st} 只，待抓 {len(codes)} 只（{DATE_TYPES[date_type]}）")
    elif not codes:
        raise ValueError("未提供股票代码")

    series = {}
    failed = {}
    success = set()
    total_rows = 0

    for idx, code in enumerate(codes, 1):
        print(f"[估值] {idx}/{len(codes)} 抓取 {code} ({DATE_TYPES[date_type]}) ...")
        try:
            rows = fetch_stock(code, date_type=date_type)
        except Exception as e:
            failed[code] = str(e)
            print(f"[估值] {code} 失败: {e}", file=sys.stderr)
            time.sleep(delay)
            continue
        if not rows:
            failed[code] = "无数据"
            print(f"[估值] {code} 无数据", file=sys.stderr)
            time.sleep(delay)
            continue
        n = ingest(rows, db_path=ingest_db)
        total_rows += n
        success.add(code)
        for code_, iname, d, *_ in rows:
            key = f"{code_}_{iname}"
            if key not in series:
                series[key] = {"n": 0, "first": d, "last": d}
            series[key]["n"] += 1
            series[key]["last"] = d
            if series[key]["n"] == 1:
                series[key]["first"] = d
        if idx % 100 == 0 or idx == len(codes):
            print(f"[估值] 进度 {idx}/{len(codes)}  成功 {len(success)}  失败 {len(failed)}  累计入库 {total_rows} 行")
        time.sleep(delay)

    failed_path = write_failed(failed)

    print("\n[估值] 入库概览")
    for key in sorted(series)[:20]:
        s = series[key]
        print(f"  {key}: n={s['n']:5d}  {s['first']} ~ {s['last']}")
    if len(series) > 20:
        print(f"  ... 共 {len(series)} 个指标系列")
    print(f"[估值] 成功 {len(success)} 只，失败 {len(failed)} 只")
    if failed_path:
        print(f"[估值] 失败清单: {failed_path}")

    result = {
        "rows_count": total_rows,
        "db_path": db_path,
        "series": series,
        "failed_path": failed_path,
        "stage_path": stage_path,
    }
    if failed:
        result["failed"] = failed
        if not series:
            raise RuntimeError(f"全部失败: {failed}")
    return result


def main():
    parser = argparse.ArgumentParser(description="东方财富估值通道抓取入库（PE/PB/PS/PCF）")
    parser.add_argument("--codes", default=None, help="逗号分隔股票代码，如 300999,601318；不传则从 stock_list 表读取全部")
    parser.add_argument("--datetype", type=int, default=1, choices=[1, 2, 3, 4],
                        help="抓取口径：1=近1年(日频,每日增量) 2=近3年(周频) 3=近5年(周频) 4=近10年(月频)，默认 1")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB 文件路径")
    parser.add_argument("--stage", default=None,
                        help="暂存 DuckDB 路径；auto 自动生成 download/temp/gzfx_stage_<时间戳>_<pid>.duckdb；不设则直入主库")
    parser.add_argument("--merge", default=None,
                        help="合并模式：暂存文件 glob，如 \"download/temp/gzfx_stage_*.duckdb\"，合并入 --db 后退出")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"每股间隔秒数，默认 {DEFAULT_DELAY}")
    parser.add_argument("--include-st", action="store_true", help="stock_list 模式下不排除 ST 股票")
    args = parser.parse_args()

    if args.merge:
        merge_stages(args.merge, db_path=args.db)
        return

    run(
        codes=args.codes,
        date_type=args.datetype,
        db_path=args.db,
        stage_path=args.stage,
        exclude_st=not args.include_st,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
