# -*- coding: utf-8 -*-
"""download 包内常用路径常量：主库、暂存目录、数据文件目录。

各 sources 模块与 runner 共用同一份路径推导，避免各处重复 os.path 拼接。
"""

import os

DOWNLOAD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(DOWNLOAD_DIR, "autots.duckdb")
TEMP_DIR = os.path.join(DOWNLOAD_DIR, "temp")
