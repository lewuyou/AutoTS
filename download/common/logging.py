# -*- coding: utf-8 -*-
"""统一日志：控制台 + 可选文件，结构化格式，按 run_id / 数据源区分。

替代原来 daily_*.py 里各自定义的 _Tee + redirect_stdout/stderr：
用标准 logging 一次配置，各数据源通过 get_logger(source, run_id) 取带上下文的 logger。
"""

import logging
import os
import sys

FORMAT = "%(asctime)s %(levelname)-7s [%(run_id)s] [%(source)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _ContextFilter(logging.Filter):
    """给每条日志补齐 run_id/source 缺省值，避免未走 get_logger 的裸 logger 报 KeyError。"""

    def filter(self, record):
        if not hasattr(record, "run_id"):
            record.run_id = "-"
        if not hasattr(record, "source"):
            record.source = "-"
        return True


def setup_logging(log_path=None, run_id="", level=logging.INFO):
    """配置根 logger：控制台 handler + 可选文件 handler。重复调用会先清空旧 handler（幂等）。

    返回根 logger。run_id 仅作缺省上下文，具体由 get_logger 覆盖。
    """
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    formatter = logging.Formatter(FORMAT, datefmt=DATE_FORMAT)
    ctx_filter = _ContextFilter()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.addFilter(ctx_filter)
    root.addHandler(console)

    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(ctx_filter)
        root.addHandler(file_handler)

    return root


class _SourceLogger(logging.LoggerAdapter):
    """为每条日志自动附带 source（数据源名）与 run_id。"""

    def __init__(self, logger, source, run_id=""):
        super().__init__(logger, {"source": source, "run_id": run_id})


def get_logger(source, run_id=""):
    """返回带 source/run_id 上下文的 logger，供各数据源模块使用。"""
    return _SourceLogger(logging.getLogger(), source, run_id)
