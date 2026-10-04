# -*- coding: utf-8 -*-
"""DuckDB 通用存储：建表、UPSERT、主库水位、暂存路径、暂存合并。

各数据源模块原来自带一份 CREATE TABLE / INSERT OR REPLACE / ATTACH-合并逻辑，
此处统一为按表名、列、建表语句参数化的通用函数。
"""

import datetime
import glob
import os

try:
    import duckdb
except ImportError:
    duckdb = None


def require_duckdb():
    if duckdb is None:
        raise ImportError("缺少 duckdb，请安装：python3 -m pip install duckdb")


def connect(db_path, read_only=False):
    """建立 DuckDB 连接（read_only=True 只读）。"""
    require_duckdb()
    return duckdb.connect(db_path, read_only=read_only)


def table_exists(con, table, catalog=None):
    """判断表是否存在；catalog 非空时用于判断 ATTACH 别名库内的表。"""
    if catalog:
        rows = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_catalog = ?",
            [catalog],
        ).fetchall()
    else:
        rows = con.execute("SELECT table_name FROM information_schema.tables").fetchall()
    return table in {r[0] for r in rows}


def create_table_if_not_exists(con, table, schema):
    """幂等建表。schema 为完整建表体，形如 '(col TYPE, ..., PRIMARY KEY (...))'。"""
    con.execute(f"CREATE TABLE IF NOT EXISTS {table} {schema}")


def upsert(con, table, columns, rows):
    """INSERT OR REPLACE 批量写入，返回写入行数；空 rows 不写。"""
    if not rows:
        return 0
    cols = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))
    con.executemany(
        f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})", rows
    )
    return len(rows)


def ingest(db_path, table, schema, columns, rows):
    """建表（幂等）+ UPSERT 批量写入，返回写入行数。

    每次调用独立连接，适合逐股/逐批落库场景（akshare 每股一次、baidu 每批一次）。
    """
    con = connect(db_path)
    try:
        create_table_if_not_exists(con, table, schema)
        return upsert(con, table, columns, rows)
    finally:
        con.close()


def max_date_by_keys(db_path, table, key_cols, date_col="date",
                     where_clause="", where_params=(), read_only=True):
    """主库增量水位：按 key_cols 分组取 date_col 最大值。

    返回 [(key_val1, ..., max_date), ...]；表不存在返回空列表。
    where_clause 形如 "adjust = ?"，where_params 为对应参数。
    """
    con = connect(db_path, read_only=read_only)
    try:
        if not table_exists(con, table):
            return []
        keys = ", ".join(key_cols)
        sql = f"SELECT {keys}, MAX({date_col}) FROM {table}"
        if where_clause:
            sql += f" WHERE {where_clause}"
        sql += f" GROUP BY {keys}"
        return con.execute(sql, list(where_params)).fetchall()
    finally:
        con.close()


def pending_stage_files(prefix, temp_dir, stage_path=None):
    """待计入增量水位的暂存库文件：本次 stage_path(存在时) + 未合并遗留 <prefix>_stage_*.duckdb。

    排除已合并（.merged 后缀）文件并去重，用于断点续抓时不重复抓取。
    """
    paths = []
    if stage_path and os.path.exists(stage_path):
        paths.append(stage_path)
    for f in glob.glob(os.path.join(temp_dir, f"{prefix}_stage_*.duckdb")):
        if f.endswith(".merged"):
            continue
        if f not in paths:
            paths.append(f)
    return paths


def max_date_by_keys_merged(db_path, stage_paths, table, key_cols, date_col="date",
                            where_clause="", where_params=()):
    """主库 + 暂存库(存在时)按 key_cols 分组取 date_col 最大值，同键取较晚。

    stage_paths 为暂存库文件路径列表（不存在的自动忽略）。
    返回与 max_date_by_keys 相同形状 [(key_val1, ..., max_date), ...]。
    恢复任务时暂存库已抓部分也要计入增量起点，避免中断重跑重复抓。
    """
    merged = {tuple(r[:-1]): r[-1] for r in max_date_by_keys(
        db_path, table, key_cols, date_col, where_clause, where_params)}
    for sp in (stage_paths or []):
        if not os.path.exists(sp):
            continue
        for r in max_date_by_keys(sp, table, key_cols, date_col, where_clause, where_params):
            k = tuple(r[:-1])
            if merged.get(k) is None or r[-1] > merged[k]:
                merged[k] = r[-1]
    return [k + (v,) for k, v in merged.items()]


def make_stage_path(prefix, temp_dir, stage_path):
    """解析暂存路径；stage_path=='auto' 时生成 <temp_dir>/<prefix>_stage_<ts>_<pid>.duckdb。"""
    if stage_path == "auto":
        os.makedirs(temp_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(temp_dir, f"{prefix}_stage_{ts}_{os.getpid()}.duckdb")
    return stage_path


def merge_stage_files(pattern, db_path, table, schema, columns):
    """把暂存 DuckDB 文件统一 UPSERT 入主库，已合并文件加 .merged 后缀。

    返回 (合并总行数, 合并文件数)。schema 用于主库建表，columns 为写入/拷贝的列。
    未找到匹配文件抛 ValueError。
    """
    files = sorted(f for f in glob.glob(pattern) if not f.endswith(".merged"))
    if not files:
        raise ValueError(f"未找到暂存文件: {pattern}")
    con = connect(db_path)
    total = 0
    merged = 0
    try:
        create_table_if_not_exists(con, table, schema)
        cols = ", ".join(columns)
        for f in files:
            con.execute(f"ATTACH '{f}' AS stage (READ_ONLY)")
            try:
                if not table_exists(con, table, catalog="stage"):
                    con.execute("DETACH stage")
                    os.rename(f, f + ".merged")
                    print(f"合并 {f}: 0 行（暂存无 {table} 表）")
                    merged += 1
                    continue
                n = con.execute(f"SELECT COUNT(*) FROM stage.{table}").fetchone()[0]
                con.execute(
                    f"INSERT OR REPLACE INTO {table} ({cols}) SELECT {cols} FROM stage.{table}"
                )
                con.execute("DETACH stage")
                os.rename(f, f + ".merged")
                total += n
                merged += 1
                print(f"合并 {f}: {n} 行")
            except Exception:
                con.execute("DETACH stage")  # 尽力清理，避免残留 ATTACH
                raise
    finally:
        con.close()
    print(f"合并完成：{merged} 个文件，共 {total} 行 -> {db_path}")
    return total, merged


def replace_table(db_path, table, schema, columns, rows):
    """快照整表重建：DROP 后按 schema 重建并批量写入（事务内），返回写入行数。

    与 ingest 的 UPSERT 不同，已在源端消失的行不会残留，适合清表重建型数据源；
    空 rows 抛 ValueError，避免异常抓取结果误清空主表。
    """
    if not rows:
        raise ValueError(f"空数据不允许重建 {table}，避免清空主表")
    con = connect(db_path)
    try:
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(f"DROP TABLE IF EXISTS {table}")
            create_table_if_not_exists(con, table, schema)
            upsert(con, table, columns, rows)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
    finally:
        con.close()
    return len(rows)


def replace_table_from_stage(pattern, db_path, table, schema, columns):
    """快照整表替换合并：用暂存表内容整体替换主库表（而非 UPSERT），已处理文件加 .merged 后缀。

    适用清表重建型数据源（如 stock_list），保证已从源端消失的行（退市股等）不残留。
    多个暂存文件按时间序依次应用，最新有效快照最终生效；
    无 {table} 表或空表的暂存文件只标记 .merged 不参与替换。
    返回 (替换行数, 处理文件数)；未找到暂存文件或无有效快照抛 ValueError。
    """
    files = sorted(f for f in glob.glob(pattern) if not f.endswith(".merged"))
    if not files:
        raise ValueError(f"未找到暂存文件: {pattern}")
    con = connect(db_path)
    total = 0
    applied = 0
    processed = 0
    try:
        cols = ", ".join(columns)
        for f in files:
            con.execute(f"ATTACH '{f}' AS stage (READ_ONLY)")
            try:
                has_table = table_exists(con, table, catalog="stage")
                n = con.execute(f"SELECT COUNT(*) FROM stage.{table}").fetchone()[0] if has_table else 0
                if n > 0:
                    con.execute("BEGIN TRANSACTION")
                    try:
                        con.execute(f"DROP TABLE IF EXISTS {table}")
                        create_table_if_not_exists(con, table, schema)
                        con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM stage.{table}")
                        con.execute("COMMIT")
                    except Exception:
                        con.execute("ROLLBACK")
                        raise
                    total = n
                    applied += 1
                    print(f"合并 {f}: 整表替换 {n} 行")
                else:
                    print(f"合并 {f}: 0 行（无有效快照，跳过）")
                con.execute("DETACH stage")
                os.rename(f, f + ".merged")
                processed += 1
            except Exception:
                con.execute("DETACH stage")  # 尽力清理，避免残留 ATTACH
                raise
    finally:
        con.close()
    if not applied:
        raise ValueError(f"暂存文件均无有效 {table} 数据: {pattern}")
    print(f"合并完成：{processed} 个文件，最新快照 {total} 行 -> {db_path}")
    return total, processed


def make_merge_stages(table, schema, columns, default_db_path):
    """返回绑定表结构的 merge_stages(pattern, db_path) 函数，供各数据源复用。

    各数据源只需声明 TABLE_NAME/TABLE_SCHEMA/COLUMNS，即可得到自己的暂存合并入口，
    避免每个源重复写一遍 3 行 merge_stages 包装。
    """
    def merge_stages(pattern, db_path=default_db_path):
        return merge_stage_files(pattern, db_path, table, schema, columns)
    return merge_stages


def make_replace_merge_stages(table, schema, columns, default_db_path):
    """快照表的 merge_stages(pattern, db_path) 绑定，语义为整表替换而非 UPSERT。"""
    def merge_stages(pattern, db_path=default_db_path):
        return replace_table_from_stage(pattern, db_path, table, schema, columns)
    return merge_stages