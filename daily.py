# -*- coding: utf-8 -*-
"""每日数据增量更新统一入口。

一个脚本完成多个数据源的日常增量抓取：暂存 + 合并 + 统一日志 + 单源隔离。
增量规则：每股/每词起点 = max(数据源最早日, 主库最大日期+1)，首次全量与每日增量同一条路径，
UPSERT 可重跑、可断点续抓；抓取期间只写暂存库，全部完成后合并入主库。

用法：
    python daily.py                       # 全部 daily 数据源（holiday 先行串行，akshare/rzrq/guzhi/tv_macro 并行抓取后串行合并；默认不含 baidu）
    python daily.py --with-baidu          # daily 数据源 + baidu（百度指数限速严格，耗时长）
    python daily.py --sources akshare     # 只跑 akshare
    python daily.py --sources akshare,baidu
    python daily.py --sources stock_list  # 手动定期刷新股票列表快照（不在默认 daily 任务中）
    python daily.py --sources nbjb        # 手动定期刷新业绩报告（季频，不在默认 daily 任务中）
    python daily.py --stage-only          # 只抓取到暂存库，不合并
    python daily.py --db 某路径.duckdb     # 指定主库
    python daily.py --end 2026-09-26      # 统一截止日期，所有数据源都截至这一天

建议 crontab（工作日早间）：
    0 9 * * 1-5  cd /Users/lwy/Code/qoder/AutoTS && .venv/bin/python daily.py >> /dev/null 2>&1

入库表（download/autots.duckdb）字段含义：

    stock_list（沪深 A 股股票列表快照，主键 symbol；手动定期触发，每次整表重建，自动剔除退市股）
        symbol          股票代码（如 "000001"）
        name            股票名称/简称（已去除空格及 C/N/S/XD 特殊前缀，ST/*ST 保留）
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

    holiday_calendar（timor.tech 节假日日历，日频，主键 date，含周末、法定节假日、调休）
        date                  日期
        is_holiday            是否放假（法定节假日 / 周末 / 调休后放假的周末）
        holiday_name          节假日名称（如 "国庆节"，普通周末/工作日为空字符串）
        is_workday_adjustment 是否调休上班日（周末上班的调休日）
        is_holiday_related    是否节假日相关日期 = is_holiday OR is_workday_adjustment

    akshare_tx（腾讯 A 股日频行情，来源 akshare stock_zh_a_hist_tx，主键 (symbol, date, adjust)）
        symbol      股票代码（6 位，抓取时自动补 sz/sh 前缀）
        date        交易日
        open        开盘价（元）
        close       收盘价（元）
        high        最高价（元）
        low         最低价（元）
        volume      成交量（股；腾讯接口原始单位为手，入库时已 ×100 换算）
        turnover    换手率（小数）
        amount      成交额（元）
        adjust      复权方式（""=不复权, qfq=前复权, hfq=后复权）

    baidu（百度指数，日频，主键 (keyword, source, date)）
        keyword  搜索关键词（股票简称，如 "金龙鱼"）
        source   指数类型：search_all=搜索指数（整体=PC+移动），feed=资讯指数
        date     日期（日频）
        value    指数值（整数，空值记 None；资讯指数自 2017-07-03 才有数据）

    rzrq（东财个股融资融券，日频，主键 (code, date)，金额单位：元；量单位：股；比率为 %；
          无两融数据股票自动记入 download/rzrq_blacklist.txt，后续运行直接跳过不发请求）
        code                 股票代码（6 位数字）
        name                 股票简称
        date                 交易日期
        margin_balance       两融余额（融资余额+融券余额，元）
        margin_balance_ratio 两融余额占流通市值比（%）
        fin_balance          融资余额（元）
        fin_balance_ratio    融资余额占流通市值比（%）
        loan_balance         融券余额（元）
        loan_balance_ratio   融券余额占流通市值比（%）
        fin_netbuy_amt       融资净买入额（融资买入-融资偿还，元）
        fin_tval_ratio       融资净买入额占成交额比（%）
        fin_buy_amt          融资买入额（元）
        fin_repay_amt        融资偿还额（元）
        loan_balance_vol     融券余量（股）
        loan_netsell_amt     融券净卖出额（元）
        loan_tval_ratio      融券净卖出额占成交额比（%）
        loan_netsell_vol     融券净卖出量（股）
        loan_sell_vol        融券卖出量（股）
        loan_repay_vol       融券偿还量（股）

    nbjb（东财业绩报告，季频，主键 (code, report_date)；全量重抓 + UPSERT 幂等，手动 --sources nbjb 触发刷新最新披露）
        code                      股票代码（6 位数字）
        name                      股票简称
        report_date               报告期截止日（如 2021-03-31 表示 2021 年一季报期末）
        report_q                  报告期季度标识（如 "2021Q1"）
        report_label              报告期中文标签（如 "2021年 一季报"）
        eps                       基本每股收益（元）
        total_operate_income      营业总收入（元，当季累计值）
        total_operate_income_yoy  营业总收入同比（%）
        parent_netprofit          归母净利润（元，当季累计值）
        parent_netprofit_yoy      归母净利润同比（%）
        notice_date               公告日期（实际披露日）

    guzhi（东财估值走势，日频，主键 (symbol, indicator, date)；与 TradingView 估值数据共用此表）
        symbol     股票代码（6 位数字）
        indicator  指标类型：pe=市盈率 pb=市净率 ps=市销率 pcf=市现率
        date       交易日期
        value      实际估值（PE/PB/PS/PCF 的 TTM 值，来自东财走势接口 INDICATOR_VALUE；空值不入库）

    tradingview_macro（TradingView 宏观品种日频 K 线，主键 (symbol, date)；品种清单见 tradingview_macro_symbols 表）
        symbol  TradingView 品种代码（如 "TVC:CN10Y"、"USDCNY"）
        name    中文名称（如 "中国10年期国债收益率"）
        date    交易日（由 UTC 时间戳转北京时间日期）
        open/high/low/close  开高低收（收益率类为百分比数值，汇率/指数为点位，期货为合约价）
        volume  成交量（股/手；指数、收益率、汇率类品种通常为空）

    tradingview_macro_symbols（TradingView 宏观品种清单配置表，主键 symbol；首次运行由 download/tv/macro_config.py 初始化）
        symbol  TradingView 品种代码（如 "TVC:CN10Y"、"USDCNY"）
        name    中文名称（如 "中国10年期国债收益率"）

新增数据源：见 download/registry.py 顶部说明。
"""

import argparse
import sys

from download import registry, runner


def main():
    parser = argparse.ArgumentParser(description="每日数据增量更新统一入口")
    parser.add_argument("--sources", default="daily",
                        help="数据源：daily（默认，=holiday,akshare,rzrq,guzhi,tv_macro）、all，或逗号分隔如 akshare,baidu；"
                             "baidu 限速严格耗时长，默认 daily 不含，需 --with-baidu 或显式 --sources baidu；"
                             "stock_list 快照表与 nbjb 季报不在默认任务中，需手动 --sources stock_list / nbjb 触发")
    parser.add_argument("--with-baidu", action="store_true",
                        help="daily 默认任务带上 baidu（百度指数，5词/批+秒级间隔，耗时长）")
    parser.add_argument("--db", default=None, help="主库 DuckDB 路径（默认 download/autots.duckdb）")
    parser.add_argument("--stage-only", action="store_true", help="只抓取到暂存库，不合并入主库")
    parser.add_argument("--end", default=None,
                        help="统一截止日期 YYYY-MM-DD，应用于所有数据源（不传则各源按自身规则：akshare=今天，baidu 按滞后规则，rzrq 增量到最新）")
    registry.add_source_args(parser)
    args = parser.parse_args()

    try:
        code = runner.run(args)
    except ValueError as e:
        print(f"参数错误: {e}", file=sys.stderr)
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
