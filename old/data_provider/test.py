import akshare as ak
df = ak.stock_zh_a_hist_tx(symbol="sz000001", start_date="20200101",
                           end_date="20260918", adjust="qfq")
# columns: date open close high low volume turnover amount