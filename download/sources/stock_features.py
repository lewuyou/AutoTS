# -*- coding: utf-8 -*-
"""股票价格衍生特征增量计算（纯库内计算，无网络抓取，不写暂存库）。

从主库 akshare_tx（后复权 hfq 行情）按每股增量计算价格衍生特征，直接 UPSERT 入主库
stock_features。作为 daily 任务最后一步：所有数据源合并完成后运行。
增量水位 = stock_features 中每股已算的最大日期，只补算其后的新交易日，重跑幂等。

新增衍生特征列：在 FEATURES 里追加 "列名" -> "SQL 表达式"（表达式作用域为 ranked
子查询，可用 r.symbol / r.date / r.high / r.low / r.close / r.prev_close 别名），
并同步在 TABLE_SCHEMA / COLUMNS 补列；历史已入库行缺少新列时需手动清表重算。

入库表 stock_features 字段含义（主键 (symbol, date)）：
    symbol          股票代码（6 位数字，如 "000001"）
    date            交易日
    amplitude       振幅（%，= (high - low) / 前收盘价 × 100；high/low 为后复权当日最高/最低，
                    前收盘价为后复权上一交易日收盘，无前收盘价的首个交易日无该行）
    log_return_1d   1 日对数收益率（小数，0.01 = +1%；= ln(当日后复权收盘 / 前一日后复权收盘)）

当日收盘或前收盘 <= 0 的行（后复权负价，约 10 只大比例分红股票的历史区间）对数收益率
无定义，跳过不入库；其余行两列均有有效值。

用法（独立运行，手动补算 / 首次全量）：
    python -m download.sources.stock_features
    python -m download.sources.stock_features --db 某路径.duckdb
"""

import argparse
import datetime
import os
import sys

from download.common import logging as common_logging
from download.common import results
from download.common import storage
from download.common.paths import DEFAULT_DB_PATH, TEMP_DIR

SOURCE_NAME = "stock_features"
TABLE_NAME = "stock_features"
TABLE_SCHEMA = """(
    symbol TEXT,
    date DATE,
    amplitude DOUBLE,
    log_return_1d DOUBLE,
    PRIMARY KEY (symbol, date)
)"""
COLUMNS = ["symbol", "date", "amplitude", "log_return_1d"]

# 衍生特征定义：列名 -> SQL 表达式（作用于 ranked 子查询，别名见 _select_sql）
FEATURES = {
    "amplitude": "(r.high - r.low) / r.prev_close * 100.0",
    "log_return_1d": "ln(r.close / r.prev_close)",
}

ADJUST = "hfq"  # 库中 akshare_tx 仅存后复权数据，衍生特征统一基于后复权价格


def _select_sql():
    """构造增量特征 SELECT：每股只取 stock_features 已算最大日期之后、且有前收盘价的行。"""
    exprs = ", ".join(f"{expr} AS {name}" for name, expr in FEATURES.items())
    return f"""
WITH ranked AS (
    SELECT symbol, date, high, low, close,
           LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS prev_close
    FROM akshare_tx
    WHERE adjust = '{ADJUST}'
),
watermark AS (
    SELECT symbol, MAX(date) AS max_date FROM {TABLE_NAME} GROUP BY symbol
)
SELECT r.symbol, r.date, {exprs}
FROM ranked r
LEFT JOIN watermark w ON w.symbol = r.symbol
WHERE r.prev_close IS NOT NULL
  AND r.prev_close > 0
  AND r.close > 0
  AND (w.max_date IS NULL OR r.date > w.max_date)
"""


def compute(db_path=DEFAULT_DB_PATH, run_id=""):
    """增量计算衍生特征并 UPSERT 入主库 stock_features，返回 RunResult。"""
    log = common_logging.get_logger(SOURCE_NAME, run_id)
    res = results.RunResult(SOURCE_NAME, run_id)
    db_path = db_path or DEFAULT_DB_PATH
    try:
        con = storage.connect(db_path)
        try:
            storage.create_table_if_not_exists(con, TABLE_NAME, TABLE_SCHEMA)
            select_sql = _select_sql()
            n = con.execute(f"SELECT COUNT(*) FROM ({select_sql}) t").fetchone()[0]
            if n:
                cols = ", ".join(COLUMNS)
                con.execute(f"INSERT OR REPLACE INTO {TABLE_NAME} ({cols}) {select_sql}")
                res.rows_merged = n
                log.info("衍生特征补算 %d 行 -> %s", n, db_path)
            else:
                log.info("衍生特征无新数据，跳过")
        finally:
            con.close()
    except Exception as e:
        log.exception("衍生特征计算失败: %s", e)
        res.fetch_status = results.FETCH_FAILED
        res.error = str(e)
        return res.finish()

    res.fetch_status = results.FETCH_OK if n else results.FETCH_NOOP
    res.merge_status = results.MERGE_MERGED
    return res.finish()


def main():
    parser = argparse.ArgumentParser(description="股票价格衍生特征增量计算")
    parser.add_argument("--db", default=None,
                        help="主库 DuckDB 路径（默认 download/autots.duckdb）")
    args = parser.parse_args()

    db_path = args.db or DEFAULT_DB_PATH
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(TEMP_DIR, exist_ok=True)
    log_path = os.path.join(TEMP_DIR, f"{SOURCE_NAME}_{run_id}.log")
    common_logging.setup_logging(log_path, run_id)
    log = common_logging.get_logger(SOURCE_NAME, run_id)
    log.info("开始，主库: %s", db_path)

    res = compute(db_path=db_path, run_id=run_id)

    log.info("结束，耗时 %s，状态 %s / %s", res.elapsed, res.fetch_status, res.merge_status)
    log.info("日志已保存: %s", log_path)
    sys.exit(0 if res.ok else 1)


if __name__ == "__main__":
    main()
