# -*- coding: utf-8 -*-
"""东方财富 行业二级分类（二级行业）板块成分抓取：刷新 stock_list.industry_l2 列（手动定期触发，整列重建）。

数据源（2026-10-12 实测）：
    板块成分股走 push2delay.eastmoney.com（push2 被 SNI 阻断后浏览器的回退节点），
    接口: GET /api/qt/clist/get?fs=b:<BK代码>&fields=f12,f14
    必须带反爬 Cookie（qgqp_b_id + nid18），否则返回 502 Bad Gateway（奇安信反爬）。
    Cookie 从真实浏览器会话提取，存 download/eastmoney_cookie.txt，失效后刷新。
    板块清单（BK代码+名称）从东财板块树 HTML 解析，HTML 路径由 --attachment 传入。

入库方式：不建独立主表，抓取结果整列刷新 stock_list.industry_l2（先清空再按 symbol 写入，
已不在任何板块的股票其二级行业随清空置 NULL）。依赖 stock_list 表已存在（先运行
python daily.py --sources stock_list）；stock_list 整表重建时会按 symbol 保留本列旧值。

断点续抓：Cookie 时效短、单次常抓不全全部板块。每抓完一个板块立即写入断点库
download/temp/industry_board_cons_state.duckdb，下次运行自动跳过已抓板块只抓剩余；
全部板块抓齐才写完整快照并整列刷新 stock_list.industry_l2（随后清空断点），
任一板块失败绝不部分刷新主库。--reset 清空断点从头抓。

用法（独立运行）：
    python -m download.sources.industry_board_cons                       # 抓取并刷新主库 stock_list.industry_l2
    python -m download.sources.industry_board_cons --stage auto          # 暂存模式
    python -m download.sources.industry_board_cons --merge "download/temp/industry_board_cons_stage_*.duckdb"

stock_list 表新增列含义：
    industry_l2  所属二级行业（东财行业二级分类板块名称，如 "银行"）
"""

import glob
import json
import os
import random
import re
import subprocess
import time

from download.common import cli
from download.common import logging as common_logging
from download.common import results
from download.common import storage
from download.common.paths import DEFAULT_DB_PATH, TEMP_DIR

SOURCE_NAME = "industry_board_cons"
# 暂存表（仅存在于暂存 DuckDB，供 merge 整列刷新 stock_list.industry_l2；非独立主表）
TABLE_NAME = "industry_board_cons"
TABLE_SCHEMA = """(
    board_code TEXT,
    board_name TEXT,
    stock_code TEXT,
    stock_name TEXT,
    PRIMARY KEY (board_code, stock_code)
)"""
COLUMNS = ["board_code", "board_name", "stock_code", "stock_name"]

STOCK_LIST_TABLE = "stock_list"
INDUSTRY_L2_COLUMN = "industry_l2"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
PROXY = "http://127.0.0.1:7897"
API = "https://push2delay.eastmoney.com/api/qt/clist/get"
COOKIE_PATH = os.path.join(os.path.dirname(__file__), "..", "eastmoney_cookie.txt")
STATE_PATH = os.path.join(TEMP_DIR, "industry_board_cons_state.duckdb")  # 断点库：已抓板块成分累积
DEFAULT_DELAY = 20
DEFAULT_ATTACHMENT = ("/Users/lwy/.qoder-cn/tmp/-Users-lwy-Code-qoder-AutoTS/attachments/"
                      "1ce69013-db16-4cc5-a782-186d8e3e0500/"
                      "2f7875a8-69b2-4332-8b70-db86e2141bba.txt")


def parse_boards(attachment_path):
    """从板块树 HTML 解析 (board_code, board_name)，去重保序。"""
    with open(attachment_path, encoding="utf-8") as f:
        html = f.read()
    pairs = re.findall(
        r"boards2-90\.(BK\d{4})\"[^>]*>.*?title=\"([^\"]+)\"", html)
    seen = {}
    for code, name in pairs:
        seen.setdefault(code, name)
    return list(seen.items())


def load_cookie():
    with open(COOKIE_PATH, encoding="utf-8") as f:
        return f.read().strip()


def _load_done_codes():
    """断点库中已抓取的板块代码集合（无断点库或表为空返回空集）。"""
    if not os.path.exists(STATE_PATH):
        return set()
    con = storage.connect(STATE_PATH, read_only=True)
    try:
        if not storage.table_exists(con, TABLE_NAME):
            return set()
        return {r[0] for r in con.execute(
            f"SELECT DISTINCT board_code FROM {TABLE_NAME}").fetchall()}
    finally:
        con.close()


def _append_board(code, name, cons):
    """把单个板块的成分股立即写入断点库（幂等 UPSERT），中断也能保留已抓部分。"""
    rows = [(code, name, sc, sn) for sc, sn in cons]
    return storage.ingest(STATE_PATH, TABLE_NAME, TABLE_SCHEMA, COLUMNS, rows)


def _snapshot_rows():
    """读出断点库全部成分股（按 COLUMNS 顺序），无断点库/无数据返回空列表。"""
    if not os.path.exists(STATE_PATH):
        return []
    con = storage.connect(STATE_PATH, read_only=True)
    try:
        if not storage.table_exists(con, TABLE_NAME):
            return []
        return con.execute(f"SELECT {', '.join(COLUMNS)} FROM {TABLE_NAME}").fetchall()
    finally:
        con.close()


def _reset_state():
    if os.path.exists(STATE_PATH):
        os.remove(STATE_PATH)


def _fetch_page(code, cookie, pn, pz=100, use_proxy=True):
    """抓单页成分股，返回 (total, [(stock_code, stock_name), ...])；失败返回 (None, [])。"""
    url = (f"{API}?pn={pn}&pz={pz}&po=1&np=1&fltt=2&invt=2&fid=f3"
           f"&fs=b:{code}&fields=f12,f14")
    cmd = ["curl", "-s", "-f"]
    if use_proxy:
        cmd += ["-x", PROXY]
    cmd += [url, "-H", f"User-Agent: {UA}", "-H", f"Cookie: {cookie}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
    if r.returncode != 0:
        return None, []
    j = json.loads(r.stdout)
    data = j.get("data") or {}
    cons = [(str(d.get("f12")), str(d.get("f14"))) for d in (data.get("diff") or [])
            if d.get("f12")]
    return data.get("total"), cons


def fetch_board(code, cookie, use_proxy=True, retries=3):
    """抓单个板块全部成分股（接口每页上限 100，按 data.total 翻页拿全）。"""
    for attempt in range(retries):
        try:
            total, cons = _fetch_page(code, cookie, pn=1, use_proxy=use_proxy)
            if total is None:
                time.sleep(1 + attempt)
                continue
            pz = 100
            page = 2
            while len(cons) < total:
                _, more = _fetch_page(code, cookie, pn=page, pz=pz, use_proxy=use_proxy)
                if not more:
                    break
                cons.extend(more)
                page += 1
                time.sleep(1.2)
            return cons
        except Exception:
            time.sleep(1 + attempt)
    return []


def _require_stock_list(con):
    if not storage.table_exists(con, STOCK_LIST_TABLE):
        raise RuntimeError(
            f"{STOCK_LIST_TABLE} 表不存在，请先运行 python daily.py --sources stock_list")


def _apply_to_stock_list(con, cons):
    """整列刷新 stock_list.industry_l2：先清空再按 symbol 写入，返回写入股票数。

    cons 为 [(stock_code, board_name), ...]。已不在任何板块的股票其二级行业随清空置 NULL。
    """
    _require_stock_list(con)
    con.execute(
        f"ALTER TABLE {STOCK_LIST_TABLE} ADD COLUMN IF NOT EXISTS {INDUSTRY_L2_COLUMN} TEXT")
    con.execute("CREATE OR REPLACE TEMP TABLE stage_cons (stock_code TEXT, board_name TEXT)")
    con.executemany("INSERT INTO stage_cons VALUES (?, ?)", cons)
    con.execute(f"UPDATE {STOCK_LIST_TABLE} SET {INDUSTRY_L2_COLUMN} = NULL")
    con.execute(
        f"UPDATE {STOCK_LIST_TABLE} SET {INDUSTRY_L2_COLUMN} = s.board_name "
        f"FROM stage_cons s WHERE {STOCK_LIST_TABLE}.symbol = s.stock_code")
    con.execute("DROP TABLE stage_cons")
    return con.execute(
        f"SELECT COUNT(*) FROM {STOCK_LIST_TABLE} WHERE {INDUSTRY_L2_COLUMN} IS NOT NULL"
    ).fetchone()[0]


def merge_stages(pattern, db_path=DEFAULT_DB_PATH):
    """暂存合并：用暂存板块成分整列刷新主库 stock_list.industry_l2，已处理文件加 .merged 后缀。

    多个暂存文件按时间序依次应用，最新有效快照最终生效；空表暂存只标记 .merged 不刷新。
    返回 (写入股票数, 处理文件数)。
    """
    files = sorted(f for f in glob.glob(pattern) if not f.endswith(".merged"))
    if not files:
        raise ValueError(f"未找到暂存文件: {pattern}")
    con = storage.connect(db_path)
    total = 0
    processed = 0
    try:
        _require_stock_list(con)
        for f in files:
            con.execute(f"ATTACH '{f}' AS stage (READ_ONLY)")
            try:
                has_table = storage.table_exists(con, TABLE_NAME, catalog="stage")
                n = con.execute(f"SELECT COUNT(*) FROM stage.{TABLE_NAME}").fetchone()[0] if has_table else 0
                if n > 0:
                    cons = con.execute(
                        f"SELECT stock_code, board_name FROM stage.{TABLE_NAME}").fetchall()
                    total = _apply_to_stock_list(con, cons)
                    print(f"合并 {f}: 刷新 {INDUSTRY_L2_COLUMN} {total} 只股票")
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
    if not total:
        raise ValueError(f"暂存文件均无有效 {TABLE_NAME} 数据: {pattern}")
    _reset_state()  # 已成功整列刷新主库，本轮断点使命完成，下次运行从头抓
    print(f"合并完成：{processed} 个文件，{STOCK_LIST_TABLE}.{INDUSTRY_L2_COLUMN} 已刷新 {total} 只股票 -> {db_path}")
    return total, processed


def run(db_path=None, stage_path=None, attachment=None, delay=DEFAULT_DELAY,
        use_proxy=True, reset=False, run_id=""):
    """执行板块成分抓取：直入主库时整列刷新 stock_list.industry_l2，暂存模式写暂存库（事后合并刷新）。

    断点续抓：已抓板块存于断点库 STATE_PATH，本次自动跳过只抓剩余；全部板块抓齐才写
    完整快照（暂存库或直入主库）并清空断点，任一板块失败只记失败清单、保留断点下次续抓。

    参数：
        db_path: 主库路径，默认 download/autots.duckdb
        stage_path: 暂存 DuckDB 路径；"auto" 自动生成，None 直入主库
        attachment: 板块树 HTML 路径，默认 DEFAULT_ATTACHMENT
        delay: 板块间隔秒数
        use_proxy: 是否走本机 7897 代理（默认 True）
        reset: True 清空断点库从头抓
    返回：
        RunResult（fetch_status/rows_staged 等）
    """
    log = common_logging.get_logger(SOURCE_NAME, run_id)
    db_path = db_path or DEFAULT_DB_PATH
    attachment = attachment or DEFAULT_ATTACHMENT
    res = results.RunResult(SOURCE_NAME, run_id)

    stage_path = storage.make_stage_path(SOURCE_NAME, TEMP_DIR, stage_path)
    if stage_path:
        log.info("暂存模式：写入 %s，事后用 merge 整列刷新 %s.%s", stage_path, STOCK_LIST_TABLE, INDUSTRY_L2_COLUMN)
    res.stage_path = stage_path

    if reset:
        _reset_state()
        log.info("已清空断点库，从头抓取")

    boards = parse_boards(attachment)
    if not boards:
        raise RuntimeError(f"板块树解析为空: {attachment}")
    log.info("解析到 %d 个二级行业板块（%s）", len(boards), attachment)
    cookie = load_cookie()

    done = _load_done_codes()
    if done:
        log.info("断点续抓：已抓 %d 个板块，跳过，剩余 %d 个", len(done), len(boards) - len(done))

    failed = {}
    fetched_this_run = 0
    for i, (code, name) in enumerate(boards, 1):
        if code in done:
            continue
        cons = fetch_board(code, cookie, use_proxy=use_proxy)
        if not cons:
            failed[code] = name
            log.warning("[%d/%d] %s(%s) 无数据/失败", i, len(boards), name, code)
        else:
            _append_board(code, name, cons)
            done.add(code)
            fetched_this_run += 1
            log.info("[%d/%d] %s(%s) %d 只", i, len(boards), name, code, len(cons))
        time.sleep(delay + random.uniform(0, 2))

    if failed:
        res.failed = len(failed)
        res.failed_path = results.failed_path_for(SOURCE_NAME, TEMP_DIR)
        results.write_failed_list(failed, res.failed_path)
        res.fetch_status = results.FETCH_PARTIAL
        res.error = (f"{len(failed)}/{len(boards)} 个板块抓取失败（已抓 {len(done)}/{len(boards)} 存断点，"
                     f"下次运行续抓剩余）: " + ",".join(failed))
        res.detail = {"boards": len(boards), "done": len(done)}
        return res.finish()

    rows = _snapshot_rows()
    if not rows:
        raise RuntimeError("抓取完成但断点库无任何成分股数据，异常")
    if stage_path:
        n = storage.ingest(stage_path, TABLE_NAME, TABLE_SCHEMA, COLUMNS, rows)
    else:
        con = storage.connect(db_path)
        try:
            n = _apply_to_stock_list(con, [(r[2], r[1]) for r in rows])
        finally:
            con.close()
        _reset_state()
    res.rows_staged = n
    res.success = len(boards)

    log.info("完成：%d 个板块全部抓齐，成分股关系 %d 条，写入 %d 行", len(boards), len(rows), n)
    res.detail = {"boards": len(boards), "cons": len(rows)}
    res.fetch_status = results.FETCH_OK
    return res.finish()


def _add_args(parser):
    parser.add_argument("--stage", default=None, help="暂存 DuckDB 路径；auto 自动生成，不设则直入主库（整列刷新 stock_list.industry_l2）")
    parser.add_argument("--attachment", default=None, help=f"板块树 HTML 路径（默认 {DEFAULT_ATTACHMENT}）")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"板块间隔秒数，默认 {DEFAULT_DELAY}")
    parser.add_argument("--no-proxy", action="store_true", help="不走本机 7897 代理")
    parser.add_argument("--reset", action="store_true", help="清空断点库从头抓取（忽略已抓板块）")


def _build_kwargs(args, db_path):
    return dict(db_path=db_path, stage_path=args.stage,
                attachment=args.attachment, delay=args.delay,
                use_proxy=not args.no_proxy, reset=args.reset)


def _log_start(log, args, db_path):
    log.info("开始，主库: %s，板块树: %s，整列刷新 stock_list.industry_l2",
             db_path, args.attachment or DEFAULT_ATTACHMENT)


def main():
    cli.standard_main(
        source_name=SOURCE_NAME,
        description="东财行业二级分类板块成分抓取（手动定期触发，整列刷新 stock_list.industry_l2）",
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
