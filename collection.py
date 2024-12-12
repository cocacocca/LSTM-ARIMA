import akshare as ak
import pandas as pd

# 股票代码
stock_code = "1A0001"
index_code = "sh000001"

# 获取大盘历史数据
index_data = ak.stock_zh_index_daily_em(symbol=index_code, start_date="19700101", end_date="20241211")
index_columns_to_keep = ["date", "open", "high", "low", "close", "volume"]  # 选择需要的列
index_renamed_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
filtered_index_data = index_data[index_columns_to_keep]
filtered_index_data.columns = index_renamed_columns

# 保存大盘数据到CSV
index_csv_name = f"{index_code}_index.csv"
filtered_index_data.to_csv(index_csv_name, index=False, encoding='utf-8')
print(f"大盘数据已保存到 {index_csv_name}")
#
# # 获取个股历史数据（原始数据和前复权数据）
# stock_data = ak.stock_zh_a_hist(symbol=stock_code, period='daily', start_date='19700101', end_date='20241210', adjust="")
# adj_close_data = ak.stock_zh_a_hist(symbol=stock_code, period='daily', start_date='19700101', end_date='20241210', adjust="qfq")
#
# # 保留需要的列并重命名
# columns_to_keep = ["日期", "开盘", "最高", "最低", "收盘", "成交量"]
# renamed_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
# filtered_stock_data = stock_data[columns_to_keep]
# filtered_stock_data.columns = renamed_columns
#
# # 添加前复权的收盘价列
# adj_close = adj_close_data["收盘"]  # 提取前复权数据中的收盘价
# filtered_stock_data["Adj_Close"] = adj_close
#
# # 保存个股数据到CSV
# stock_csv_name = f"{stock_code}.csv"
# filtered_stock_data.to_csv(stock_csv_name, index=False, encoding='utf-8')
# print(f"个股数据已保存到 {stock_csv_name}")
