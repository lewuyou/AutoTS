# -*- coding: utf-8 -*-
"""统一任务结果格式与失败清单写入。

各数据源 run() 返回 RunResult，供统一入口汇总与决定退出码；
抓取状态与合并状态分开记录。
"""

import datetime
import os

# 抓取状态
FETCH_OK = "success"        # 计划内全部抓取成功（含无新数据）
FETCH_PARTIAL = "partial"   # 部分失败（存在真实失败项）
FETCH_FAILED = "failed"     # 全部失败
FETCH_NOOP = "noop"         # 无需抓取（库内均已覆盖到结束日）

# 合并状态
MERGE_MERGED = "merged"     # 暂存已合并入主库
MERGE_PENDING = "pending"   # 暂存已生成但未合并（主库被占用等）
MERGE_SKIPPED = "skipped"   # 无暂存文件，无需合并


class RunResult:
    """单个数据源一次运行的结果。"""

    def __init__(self, source, run_id=""):
        self.source = source
        self.run_id = run_id
        self.started_at = datetime.datetime.now()
        self.finished_at = None
        self.fetch_status = FETCH_OK
        self.merge_status = MERGE_SKIPPED
        self.rows_staged = 0    # 写入暂存库行数
        self.rows_merged = 0    # 合并入主库行数
        self.success = 0        # 成功抓取的序列/关键词/股票数
        self.failed = 0         # 失败项数
        self.no_new = 0         # 无新数据项数
        self.skipped = 0        # 已是最新被跳过项数
        self.stage_path = None
        self.failed_path = None
        self.error = None       # 整体异常信息（fetch_status=FAILED 时）
        self.detail = {}        # 数据源自定义字段

    @property
    def elapsed(self):
        end = self.finished_at or datetime.datetime.now()
        return end - self.started_at

    def finish(self):
        self.finished_at = datetime.datetime.now()
        return self

    @property
    def ok(self):
        """整体是否算成功：抓取无真实失败，且不处于待合并状态。"""
        return self.fetch_status in (FETCH_OK, FETCH_NOOP) and self.merge_status != MERGE_PENDING

    def to_dict(self):
        return {
            "source": self.source,
            "run_id": self.run_id,
            "fetch_status": self.fetch_status,
            "merge_status": self.merge_status,
            "rows_staged": self.rows_staged,
            "rows_merged": self.rows_merged,
            "success": self.success,
            "failed": self.failed,
            "no_new": self.no_new,
            "skipped": self.skipped,
            "stage_path": self.stage_path,
            "failed_path": self.failed_path,
            "elapsed_seconds": round(self.elapsed.total_seconds(), 1),
            "detail": self.detail,
        }


def failed_path_for(prefix, temp_dir, ts=None):
    """生成失败清单路径 <temp_dir>/<prefix>_failed_<ts>.txt。"""
    ts = ts or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(temp_dir, f"{prefix}_failed_{ts}.txt")


def write_failed_list(rows, path):
    """写失败清单（tab 分隔：key + 原因），返回写入项数。"""
    if not rows:
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for key in sorted(rows):
            f.write(f"{key}\t{rows[key]}\n")
    return len(rows)
