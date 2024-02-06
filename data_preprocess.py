# -*- coding:UTF-8 -*-

"""
文件说明：
"""

import pyarrow.parquet as pq
from sqlalchemy import create_engine
import math
import warnings
import numpy as np
import time
import pandas as pd
from scipy.stats import norm
warnings.simplefilter('ignore')
import sys
sys.path.append(r"C:\Users\Sendoh\Desktop\data")
import os
import sys
import datetime

M = 4
rf = 3.10 * 0.01

class base_data():
    def __init__(self):
        self.mongo_username = "zlt01"
        self.mongo_password = "zlt_ujYH"
        self.mongo_host = "mongodb://zlt01:zlt_ujYH@192.168.9.189:15009/data"

        self.database85_username = 'chuangXin'
        self.database85_password = 'Xini.100'
        self.database85_host = '192.168.9.85'
        self.database85_basename = 'option_new'

        self.wind_username = 'quantchina'
        self.wind_password = 'zMxq7VNYJljTFIQ8'
        self.wind_host = '192.168.7.93'
        self.wind_port = 3306

        self.data_path = "//192.168.7.92/data/"

    def get_tick_data(self, symbol, date, symbol_type="stock"):
        """
        从datahouse中获取tick，期权的数据在股票类里
        :param symbol: str,format like IF2109.CFE or 600001.SH
        :param date : str,format like '20210907'
         param symbol_type : str, must be 'future' or 'stock'
        :return: dataframe
        """
        df = pd.DataFrame()
        try:
            code = (symbol.split('.')[0]).lower()
            exchange = (symbol.split('.')[1]).lower()
        except Exception as e:
            print(e)
            return df
        if symbol_type == "future" or symbol_type == "stock":
            tick_dir = self.data_path + "/tick/" + symbol_type + "/" + date + "/quote/"
            file_name = exchange + "_" + code + "_" + date + "_quote.parquet"
            if os.path.exists(tick_dir + file_name):
                df = pd.read_parquet(tick_dir + file_name)
            else:
                print("tick data path error:", tick_dir + file_name)
        return df

    def get_minbar_data(self, symbol, date, symbol_type="future"):
        """
        从datahouse中获取分钟bar，期权的数据在股票类里
        :param symbol: str,format like IF2109.CFE or 600001.SH
        :param date : str,format like '20210907'
        :param symbol_type : str, must be 'future' or 'stock'
        :return: dataframe
        """
        df = pd.DataFrame()
        try:
            code = (symbol.split('.')[0]).lower()
            exchange = (symbol.split('.')[1]).lower()
        except Exception as e:
            print(e)
            return df
        if symbol_type == "future" or symbol_type == "stock":
            min_bar_dir = self.data_path + "/minbar/" + symbol_type + "/" + date + "/1min/"
            file_name = exchange + "_" + code + "_" + date + "_1min.parquet"
            if os.path.exists(min_bar_dir + file_name):
                df = pd.read_parquet(min_bar_dir + file_name)
            else:
                print("min bar path error:", min_bar_dir + file_name)
        return df

    def get_csv_summary(self, head, date, symbol_type="future"):
        df = pd.DataFrame()
        summary_dir = self.data_path + "/tick//" + symbol_type + "//" + date + "/"
        file_name = head + "_quote_summary.csv"
        if os.path.exists(summary_dir + file_name):
            df = pd.read_csv(summary_dir + file_name)
        else:
            print("tick data path error:", summary_dir + file_name)
        return df


def get_option_minbar(date, symbol):
    bd = base_data()
    data = bd.get_tick_data(symbol, date, symbol_type="future")
    data.index = pd.to_datetime(data['datetime'])
    data['acc_volume'] = data['volume']
    data['volume'] = data['volume'].diff().fillna(method='bfill')
    data['acc_turnover'] = data['turnover']
    data['turnover'] = data['turnover'].diff().fillna(method='bfill')
    data['trade_price'] = data['turnover'] / data['volume']

    df_symbol = data['symbol'].resample('1min', label='left', closed='left').last()
    df_open = data['last_prc'].resample('1min', label='left', closed='left').first()
    df_high = data['last_prc'].resample('1min', label='left', closed='left').max()
    df_low = data['last_prc'].resample('1min', label='left', closed='left').min()
    df_close = data['last_prc'].resample('1min', label='left', closed='left').last()
    df_open_interest = data['open_interest'].resample('1min', label='left', closed='left').last()
    df_volume = data['volume'].resample('1min', label='left', closed='left').sum()
    df_acc_volume = data['acc_volume'].resample('1min', label='left', closed='left').last()
    df_turnover = data['turnover'].resample('1min', label='left', closed='left').sum()
    df_acc_turnover = data['acc_turnover'].resample('1min', label='left', closed='left').last()
    df_last_ask_prc1 = data['ask_prc1'].resample('1min', label='left', closed='left').last()
    df_last_ask_vol1 = data['ask_vol1'].resample('1min', label='left', closed='left').last()
    df_last_bid_prc1 = data['bid_prc1'].resample('1min', label='left', closed='left').last()
    df_last_bid_vol1 = data['bid_vol1'].resample('1min', label='left', closed='left').last()
    df_first_ask_prc1 = data['ask_prc1'].resample('1min', label='left', closed='left').first()
    df_first_ask_vol1 = data['ask_vol1'].resample('1min', label='left', closed='left').first()
    df_first_bid_prc1 = data['bid_prc1'].resample('1min', label='left', closed='left').first()
    df_first_bid_vol1 = data['bid_vol1'].resample('1min', label='left', closed='left').first()

    Resample_data = pd.concat([df_symbol, df_open, df_high, df_low, df_close, df_open_interest, df_volume, df_acc_volume,
                               df_turnover, df_acc_turnover, df_last_ask_prc1, df_last_ask_vol1,
                               df_last_bid_prc1, df_last_bid_vol1, df_first_ask_prc1, df_first_ask_vol1,
                               df_first_bid_prc1, df_first_bid_vol1], axis=1)
    Resample_data.columns = ["symbol", "open", "high", "low", "close", "open_interest", "volume", "acc_volume", "turnover",
                             "acc_turnover",
                             'last_ask_prc1', 'last_ask_vol1', 'last_bid_prc1', 'last_bid_vol1',
                             'first_ask_prc1', 'first_ask_vol1', 'first_bid_prc1', 'first_bid_vol1']

    Resample_data = Resample_data.reset_index()

    Resample_data['date'] = Resample_data['datetime'].apply(lambda x: x.date())
    Resample_data['time'] = Resample_data['datetime'].apply(lambda x: x.time())

    t0 = pd.to_datetime('21:00:00').time()
    t1 = pd.to_datetime('02:30:00').time()
    t2 = pd.to_datetime('09:00:00').time()
    t3 = pd.to_datetime('10:15:00').time()
    t4 = pd.to_datetime('10:30:00').time()
    t5 = pd.to_datetime('11:30:00').time()
    t6 = pd.to_datetime('13:30:00').time()
    t7 = pd.to_datetime('15:00:00').time()

    DATA = Resample_data.set_index('time')
    d1 = DATA[t0:t1]
    d2 = DATA[t2:t3]
    d3 = DATA[t4:t5]
    d4 = DATA[t6:t7]

    W_data = pd.concat([d1, d2, d3, d4], axis=0)
    W_data = W_data.fillna(method='ffill').reset_index()

    # 夜盘属于第二天
    W_data['trading_date'] = W_data.apply(lambda x: x['date'] + datetime.timedelta(days=1) if x['time'] >= t0 else x['date'], axis=1)

    W_data = W_data[
        ["symbol", "trading_date", "datetime", "date", "time", "open", "high", "low", "close", "open_interest", "volume", "acc_volume", "turnover",
         "acc_turnover",
         'last_ask_prc1', 'last_ask_vol1', 'last_bid_prc1', 'last_bid_vol1',
         'first_ask_prc1', 'first_ask_vol1', 'first_bid_prc1', 'first_bid_vol1']]
    return W_data


def Get_new_all_minbar_data(date, head, symbol_type="future", update: bool = False):
    """
    :param date: '20220630'
    :param head: "ine"
    :param symbol_type: "future"
    :return: combined dataframe
    """
    bd = base_data()
    summary_data = bd.get_csv_summary(head, date, symbol_type)
    summary_data['symbol_low'] = summary_data['symbol'].apply(lambda x: x.lower())
    summary_data['file name'] = summary_data['symbol_low'].apply(
        lambda x: x.split('.')[1] + "_" + x.split('.')[0] + "_" + date + "_quote.parquet")
    file_list = list(summary_data['symbol'])

    folder_path1 = f'C:/Users/Sendoh/Desktop/data/oil/minbar/individual/{date}'
    if update or not os.path.exists(folder_path1):
        os.makedirs(folder_path1)
        print("路径1已创建")

    folder_path2 = f'C:/Users/Sendoh/Desktop/data/oil/minbar/combined/{date}'
    if update or not os.path.exists(folder_path2):
        os.makedirs(folder_path2)
        print("路径2已创建")


    Minbar_all_data = pd.DataFrame()
    count = 0
    for file in file_list:
        symbol = file
        temp_minbar_data = get_option_minbar(date, symbol)
        file_path1 = f'C:/Users/Sendoh/Desktop/data/oil/minbar/individual/{date}/{symbol}.{date}_parquet'
        temp_minbar_data.to_parquet(file_path1)
        #temp_minbar_data.to_parquet(f'D:\CSC\ALL CODE\Quant\期权期货\已处理数据\wash_data_myself\minbar_data\individual\\{date}\\{symbol}.{date}_parquet')
        count += 1
        print(count)
        Minbar_all_data = pd.concat([Minbar_all_data, temp_minbar_data], axis=0)

    file_path2 = f'C:/Users/Sendoh/Desktop/data/oil/minbar/combined/{date}/{head}.{date}_parquet'
    Minbar_all_data.to_parquet(file_path2)

    return Minbar_all_data


date = '20221213'
head = 'ine'
symbol_type = 'future'
All_minbar_data = Get_new_all_minbar_data(date, head, symbol_type)

