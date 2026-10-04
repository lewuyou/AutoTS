# -*- coding: utf-8 -*-
"""A 股股票列表抓取入库模块。

通过 akshare 的 stock_info_sh_name_code（上交所）和 stock_info_sz_name_code（深交所）
抓取所有沪深 A 股股票代码和名称，入库到 DuckDB 的 stock_list 表，供其他模块筛选股票使用。

快照表：不参与 download_all.py 默认批量任务，仅手动 --source stock_list 触发；
每次运行先清表再全量重建（自动剔除退市股）。

可独立运行，也可由 download_all.py 调用 run()。

用法（独立运行）：
    python -m download.stock_list              # 抓取全部 A 股股票列表

依赖：
    pip install akshare duckdb pandas

入库表 stock_list 字段含义：
    symbol          股票代码（如 "000001"）
    name            股票名称/简称（如 "平安银行"；已去除空格及 C/N/S/XD 特殊前缀，ST/*ST 保留）
    exchange        交易所（SH=上交所, SZ=深交所）
    full_name       证券全称（仅 SH）
    company_short   公司简称（仅 SH）
    company_full    公司全称（仅 SH）
    list_date       上市日期
    board           板块（SH: 主板/科创板, SZ: 主板/创业板）
    total_share     A股总股本（单位：股，SZ 来自接口，SH 由市值/最新价推算）
    float_share     A股流通股本（单位：股，SZ 来自接口，SH 由市值/最新价推算）
    industry        所属行业（仅 SZ）
    total_mv        总市值（单位：元，来自 stock_zh_a_spot_tx）
    float_mv        流通市值（单位：元，来自 stock_zh_a_spot_tx）
    last_price      最新价（单位：元，来自 stock_zh_a_spot_tx）
"""

import argparse
import os
import sys

try:
    import akshare as ak
except ImportError:
    ak = None

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    import pandas as pd
except ImportError:
    pd = None

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autots.duckdb")
TABLE_NAME = "stock_list"
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_list.csv")


def _require_akshare():
    if ak is None:
        raise ImportError("缺少 akshare，请安装：python3 -m pip install akshare")


def _require_duckdb():
    if duckdb is None:
        raise ImportError("缺少 duckdb，请安装：python3 -m pip install duckdb")


def _require_pandas():
    if pd is None:
        raise ImportError("缺少 pandas，请安装：python3 -m pip install pandas")


def _parse_share(value):
    """解析股本字符串（如 "19,405,918,198"）为整数，失败返回 None。"""
    if value is None:
        return None
    try:
        return int(str(value).replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _s(value):
    """转字符串，NaN/空值返回 None（避免 pandas NaN 被转成字符串 "nan" 入库）。"""
    if value is None or (pd is not None and pd.isna(value)):
        return None
    return str(value)


def clean_stock_name(name, company_short=None):
    """清理股票简称：去空格，去除特殊前缀标识，返回可用于搜索的完整简称。

    上交所证券简称有 4 字上限，除权(XD)/上市首日(N)/新股(C)/未股改(S)等临时前缀
    会挤占长度使主体被截断（如 "XD三元股" → 主体只剩 "三元股"）。此时改用公司简称
    字段取完整名（如 "三元股份"）。
    ST/*ST 为退市风险标识，按约定保留不处理；TCL 等真实字母缩写不是标识，保留。
    """
    def _strip(raw):
        if raw is None:
            return None
        raw = str(raw).replace(" ", "").strip()
        for prefix in ("XD", "N", "C"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
                break
        # S 仅指未股改标识（后接中文简称），需排除 ST/*ST
        if raw.startswith("S") and not raw.startswith("ST"):
            raw = raw[1:]
        return raw

    if name is None:
        return None
    raw = str(name).replace(" ", "").strip()
    has_prefix = raw.startswith(("XD", "N", "C")) or (raw.startswith("S") and not raw.startswith("ST"))
    if has_prefix and company_short is not None:
        base = _strip(company_short)
        if base:
            return base
    return _strip(raw)


def fetch_stock_list():
    """通过 akshare 拉取沪深 A 股股票列表，返回 dict 列表。

    使用 stock_info_sh_name_code（上交所）和 stock_info_sz_name_code（深交所），
    相同信息放同名字段，不同信息单独列字段。
    市值和最新价通过 stock_zh_a_spot_tx 一次性获取并合并。
    SH 的股本由市值/最新价推算。
    """
    _require_akshare()

    stocks = []

    # 上交所主板
    df_sh = ak.stock_info_sh_name_code(symbol="主板A股")
    for _, row in df_sh.iterrows():
        stocks.append({
            "symbol": str(row["证券代码"]).zfill(6),
            "name": clean_stock_name(row["证券简称"], company_short=row["公司简称"]),
            "exchange": "SH",
            "full_name": _s(row["证券全称"]),
            "company_short": _s(row["公司简称"]),
            "company_full": _s(row["公司全称"]),
            "list_date": row["上市日期"],
            "board": "主板",
            "total_share": None,
            "float_share": None,
            "industry": None,
            "total_mv": None,
            "float_mv": None,
            "last_price": None,
        })

    # 上交所科创板
    df_kcb = ak.stock_info_sh_name_code(symbol="科创板")
    for _, row in df_kcb.iterrows():
        stocks.append({
            "symbol": str(row["证券代码"]).zfill(6),
            "name": clean_stock_name(row["证券简称"], company_short=row["公司简称"]),
            "exchange": "SH",
            "full_name": _s(row["证券全称"]),
            "company_short": _s(row["公司简称"]),
            "company_full": _s(row["公司全称"]),
            "list_date": row["上市日期"],
            "board": "科创板",
            "total_share": None,
            "float_share": None,
            "industry": None,
            "total_mv": None,
            "float_mv": None,
            "last_price": None,
        })

    # 深交所
    df_sz = ak.stock_info_sz_name_code()
    for _, row in df_sz.iterrows():
        stocks.append({
            "symbol": str(row["A股代码"]).zfill(6),
            "name": clean_stock_name(row["A股简称"]),
            "exchange": "SZ",
            "full_name": None,
            "company_short": None,
            "company_full": None,
            "list_date": row["A股上市日期"],
            "board": _s(row["板块"]),
            "total_share": _parse_share(row["A股总股本"]),
            "float_share": _parse_share(row["A股流通股本"]),
            "industry": _s(row["所属行业"]),
            "total_mv": None,
            "float_mv": None,
            "last_price": None,
        })

    # 用 stock_zh_a_spot_tx 补齐市值和最新价（腾讯源，稳定且包含北交所）
    print("[StockList] 获取全市场市值数据 ...")
    df_spot = ak.stock_zh_a_spot_tx()
    spot_map = {}
    for _, row in df_spot.iterrows():
        # 去掉 sh/sz 前缀，统一为 6 位代码
        code = str(row["code"]).replace("sh", "").replace("sz", "").zfill(6)
        try:
            last_price = float(row["zxj"]) if row["zxj"] else None
        except (ValueError, TypeError):
            last_price = None
        try:
            # zsz 单位是亿元，转换为元
            total_mv = float(row["zsz"]) * 100000000 if row["zsz"] else None
        except (ValueError, TypeError):
            total_mv = None
        try:
            # ltsz 单位是亿元，转换为元
            float_mv = float(row["ltsz"]) * 100000000 if row["ltsz"] else None
        except (ValueError, TypeError):
            float_mv = None
        spot_map[code] = {
            "total_mv": total_mv,
            "float_mv": float_mv,
            "last_price": last_price,
        }

    for s in stocks:
        code = s["symbol"]
        if code in spot_map:
            s["total_mv"] = spot_map[code]["total_mv"]
            s["float_mv"] = spot_map[code]["float_mv"]
            s["last_price"] = spot_map[code]["last_price"]

    # 股本推算：市值 / 最新价（SH 全部推算，SZ 优先用接口数据，缺失时推算）
    for s in stocks:
        if s["last_price"] and s["last_price"] > 0:
            if s["total_mv"] and not s["total_share"]:
                s["total_share"] = int(s["total_mv"] / s["last_price"])
            if s["float_mv"] and not s["float_share"]:
                s["float_share"] = int(s["float_mv"] / s["last_price"])

    return stocks, df_sh, df_kcb, df_sz, df_spot


def save_csv(df_sh, df_kcb, df_sz, df_spot, csv_path=CSV_PATH):
    """保存原始数据为 CSV 文件。"""
    _require_pandas()
    # 合并所有原始数据，添加来源标记
    df_sh["source"] = "sh_main"
    df_kcb["source"] = "sh_kcb"
    df_sz["source"] = "sz"
    df_spot["source"] = "spot_tx"
    
    # 保存到不同 sheet 或合并保存
    with pd.ExcelWriter(csv_path.replace(".csv", ".xlsx")) as writer:
        df_sh.to_excel(writer, sheet_name="sh_main", index=False)
        df_kcb.to_excel(writer, sheet_name="sh_kcb", index=False)
        df_sz.to_excel(writer, sheet_name="sz", index=False)
        df_spot.to_excel(writer, sheet_name="spot_tx", index=False)
    
    return csv_path.replace(".csv", ".xlsx")


def ingest(rows, db_path=DEFAULT_DB_PATH):
    """清表后全量重建 stock_list（快照表，每次运行即全量刷新，自动剔除退市股）。"""
    _require_duckdb()
    con = duckdb.connect(db_path)
    try:
        con.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        con.execute(
            f"""
            CREATE TABLE {TABLE_NAME} (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                exchange TEXT,
                full_name TEXT,
                company_short TEXT,
                company_full TEXT,
                list_date DATE,
                board TEXT,
                total_share BIGINT,
                float_share BIGINT,
                industry TEXT,
                total_mv DOUBLE,
                float_mv DOUBLE,
                last_price DOUBLE
            )
            """
        )
        con.executemany(
            f"""
            INSERT INTO {TABLE_NAME}
            (symbol, name, exchange, full_name, company_short, company_full,
             list_date, board, total_share, float_share, industry, total_mv, float_mv, last_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["symbol"], r["name"], r["exchange"],
                    r["full_name"], r["company_short"], r["company_full"],
                    r["list_date"], r["board"], r["total_share"],
                    r["float_share"], r["industry"], r["total_mv"], r["float_mv"],
                    r["last_price"],
                )
                for r in rows
            ],
        )
    finally:
        con.close()
    return len(rows)


def run(db_path=None, csv_path=None):
    """供外部调用的入口。

    参数：
        db_path: DuckDB 文件路径，默认模块目录下 autots.duckdb
        csv_path: CSV 文件路径，默认模块目录下 stock_list.csv
    返回：
        dict 包含 rows_count, db_path, csv_path, exchanges 等
    """
    db_path = db_path or DEFAULT_DB_PATH
    csv_path = csv_path or CSV_PATH

    print("[StockList] 抓取全部 A 股股票列表 ...")
    try:
        stocks, df_sh, df_kcb, df_sz, df_spot = fetch_stock_list()
    except Exception as e:
        print(f"[StockList] 失败: {e}", file=sys.stderr)
        raise

    if not stocks:
        raise RuntimeError("无数据")

    # 保存原始数据
    raw_path = save_csv(df_sh, df_kcb, df_sz, df_spot, csv_path=csv_path)
    print(f"[StockList] 原始数据已保存: {raw_path}")

    # 入库
    n = ingest(stocks, db_path=db_path)

    exchanges = {}
    for r in stocks:
        ex = r["exchange"]
        exchanges[ex] = exchanges.get(ex, 0) + 1

    print(f"[StockList] 入库 {n} 行")
    print("\n[StockList] 交易所分布")
    for ex in sorted(exchanges):
        print(f"  {ex}: {exchanges[ex]} 只")

    return {
        "rows_count": n,
        "db_path": db_path,
        "csv_path": raw_path,
        "exchanges": exchanges,
    }


def main():
    parser = argparse.ArgumentParser(description="A 股股票列表抓取入库")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB 文件路径")
    parser.add_argument("--csv", default=CSV_PATH, help="CSV 文件路径")
    args = parser.parse_args()

    run(db_path=args.db, csv_path=args.csv)


if __name__ == "__main__":
    main()
