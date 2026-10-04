# -*- coding: utf-8 -*-
"""download/sources 数据源接入模块。

每个数据源一个文件，暴露统一契约：
    TABLE_NAME / TABLE_SCHEMA / COLUMNS   表结构声明（建表 + UPSERT 用）
    run(**kw) -> RunResult                抓取 + 暂存（不在此合并）
    merge_stages(pattern, db_path)        暂存合并入主库
    main()                                argparse CLI（可独立运行）

公共流程（日志/建表/UPSERT/水位/暂存路径/合并/结果）统一走 download.common，
这里只保留各数据源自身的抓取策略。
"""
