# -*- coding:UTF-8 -*-

"""
文件说明：
对于单张期权在某一时刻的压力测试 ________原油
"""

import math
import warnings
import pandas as pd
from scipy.stats import norm

warnings.simplefilter('ignore')
# import sys
# sys.path.append(r'D:\CSC\ALL CODE\Quant\期权期货\raw data')
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import matplotlib.ticker as ticker

a = pd.read_parquet("C:/Users/Sendoh/Desktop/data/oil/minbar/individual/20221213/SC2302P430.INE.20221213_parquet")

M = 4
rf = 3.10 * 0.01


# 期权定价
def futOptPrice(S0, K, T, r, sigma, option_type):
    """
    :param S0: 标的现在价格
    :param K:  行权价格
    :param T:  到期时间：以年为单位
    :param r:  利率
    :param sigma:  波动率
    :param option_type:  期权类型：C or P
    :return:   期权价格
    """
    d1 = (math.log(S0 / K) + (r + sigma * sigma / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    c = math.exp(-r * T) * (S0 * norm.cdf(d1) - K * norm.cdf(d2))
    p = math.exp(-r * T) * (K * norm.cdf(-d2) - S0 * norm.cdf(-d1))

    if option_type == 'C':
        return c
    else:
        return p


# 隐含波动率
def impv(S0, K, T, r, op_value_market, option_type):
    """
   :param S0: 标的现在价格
    :param K:  行权价格
    :param T:  到期时间：以年为单位
    :param r:  利率
    :param op_value_market: 期权现在的市场价格
    :param option_type: 期权类型
    :return:
    """
    if S0 == None or S0 == 0 or op_value_market == None or op_value_market == 0:
        iv = None
    else:
        c_func = lambda sigma: futOptPrice(S0, K, T, r, sigma, option_type) - op_value_market
        iv_value = fsolve(c_func, np.array([0.3]), xtol=0.000001)
        iv = iv_value[0]
        if iv <= 0:
            iv = None
    return iv


# 组合
class Greeks:
    def __init__(self, S0, K, T, r, sigma, option_type):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.d1 = (math.log(S0 / K) + (r + sigma * sigma / 2) * T) / (sigma * math.sqrt(T))
        self.d2 = self.d1 - sigma * math.sqrt(T)

    def Delta(self):
        q = 0  # q = b - r
        d1_v = self.d1
        if self.option_type == 'C':
            delta = np.exp(-q * self.T) * norm.cdf(d1_v)
        else:
            delta = np.exp(-q * self.T) * (norm.cdf(d1_v) - 1)
        return delta * self.S0 / 100

    def Gamma(self):
        q = 0
        d1_v = self.d1
        a = norm.pdf(d1_v) * np.exp(-q * self.T)
        b = self.S0 * self.sigma * np.sqrt(self.T)
        gamma = a / b * self.S0 / 100  # 当标的价格变化1%， delta变化量
        delta_gamma = gamma * self.S0 / 100  # 当标的变化1%，delta变化量带来的期权价格变化
        return delta_gamma

    def Vega(self):
        q = 0
        d1_v = self.d1
        return self.S0 * np.sqrt(self.T) * norm.pdf(d1_v) * np.exp(-q * self.T) / 100 * self.sigma

    # 时间减少一天，期权价格的衰退值
    def Theta(self):
        q = 0
        d1_v = self.d1
        d2_v = self.d2
        if self.option_type == 'C':
            a = -self.S0 * norm.pdf(d1_v) * self.sigma * np.exp(-q * self.T) / (2 * np.sqrt(self.T))
            b = q * self.S0 * norm.cdf(d1_v) * np.exp(-q * self.T)
            c = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2_v)
            theta = a + b - c
        else:
            a = -self.S0 * norm.pdf(-d1_v) * self.sigma * np.exp(-q * self.T) / (2 * np.sqrt(self.T))
            b = q * self.S0 * norm.cdf(-d1_v) * np.exp(-q * self.T)
            c = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2_v)
            theta = a - b + c
        return theta / 365

    # 期权价格关于无风险利率变化百分之一是变化的量
    def rho(self):
        q = 0
        d2_v = self.d2
        if self.option_type == 'C':
            rho = self.K * self.T * np.exp(self.r * self.T) * norm.cdf(d2_v)
        else:
            rho = -self.K * self.T * np.exp(self.r * self.T) * norm.cdf(-d2_v)
        return rho / 100


# 压力测试: 隐含波动率变化
def IV_variation(S0, K, T, r, op_value_market, option_type, long_short):
    unit = 1000
    T = T / 365
    old_iv = impv(S0, K, T, r, op_value_market, option_type)
    old_greeks = Greeks(S0, K, T, r, old_iv, option_type)
    PM_IV_information = pd.DataFrame(
        [[f"old iv = {old_iv}", op_value_market, op_value_market, S0, 0, old_iv, old_greeks.Delta(),
          old_greeks.Delta() * unit,
          old_greeks.Gamma(), old_greeks.Gamma() * unit, old_greeks.Vega(), old_greeks.Vega() * unit,
          old_greeks.Theta(), old_greeks.Theta() * unit, old_greeks.rho(), old_greeks.rho() * unit]],
        columns=['IV变化', '当前期权价格', '期权价格', '标的价格', '期权价格变化率', 'iv', 'Delta', 'Delta Cash', 'Gamma', 'Gamma Cash',
                 'Vega',
                 'Vega Cash', 'Theta', 'Theta Cash', 'rho', 'rho Cash'])

    iv_c = 0.1
    while iv_c < 1:
        change = f"IV等于{iv_c}"
        op_value_c = futOptPrice(S0, K, T, r, iv_c, option_type)
        greeks_c = Greeks(S0, K, T, r, iv_c, option_type)
        price_change_prc = (op_value_c - op_value_market) / op_value_market
        IF_c = pd.DataFrame(
            [[change, op_value_market, op_value_c, S0, price_change_prc, iv_c, greeks_c.Delta(),
              greeks_c.Delta() * unit,
              greeks_c.Gamma(),
              greeks_c.Gamma() * unit, greeks_c.Vega(), greeks_c.Vega() * unit,
              greeks_c.Theta(), greeks_c.Theta() * unit, greeks_c.rho(), greeks_c.rho() * unit]],
            columns=['IV变化', '当前期权价格', '期权价格', '标的价格', '期权价格变化率', 'iv', 'Delta', 'Delta Cash', 'Gamma', 'Gamma Cash',
                     'Vega',
                     'Vega Cash', 'Theta', 'Theta Cash', 'rho', 'rho Cash'])
        PM_IV_information = pd.concat([IF_c, PM_IV_information], axis=0)
        iv_c = iv_c + 0.05

    if long_short == 'long':
        PM_IV_information = PM_IV_information
    elif long_short == 'short':
        PM_IV_information['当前期权价格'] = -PM_IV_information['当前期权价格']
        PM_IV_information['期权价格'] = -PM_IV_information['期权价格']
        PM_IV_information['Delta'] = -PM_IV_information['Delta']
        PM_IV_information['Delta Cash'] = -PM_IV_information['Delta Cash']
        PM_IV_information['Gamma'] = -PM_IV_information['Gamma']
        PM_IV_information['Gamma Cash'] = -PM_IV_information['Gamma Cash']
        PM_IV_information['Vega'] = -PM_IV_information['Vega']
        PM_IV_information['Vega Cash'] = -PM_IV_information['Vega Cash']
        PM_IV_information['Theta'] = -PM_IV_information['Theta']
        PM_IV_information['Theta Cash'] = -PM_IV_information['Theta Cash']
        PM_IV_information['rho'] = -PM_IV_information['rho']
        PM_IV_information['rho Cash'] = -PM_IV_information['rho Cash']

    return PM_IV_information.dropna()


# 标的价格变化
def Target_price_variation(S0, K, T, r, op_value_market, option_type, long_short):
    unit = 1000
    T = T / 365
    old_iv = impv(S0, K, T, r, op_value_market, option_type)
    old_greeks = Greeks(S0, K, T, r, old_iv, option_type)
    PM_IV_information = pd.DataFrame(
        [["0", op_value_market, op_value_market, S0, 0, 0, old_iv, old_greeks.Delta(), old_greeks.Delta() * unit,
          old_greeks.Gamma(), old_greeks.Gamma() * unit, old_greeks.Vega(), old_greeks.Vega() * unit,
          old_greeks.Theta(), old_greeks.Theta() * unit, old_greeks.rho(), old_greeks.rho() * unit]],
        columns=['标的价格变化', '当前期权价格', '期权价格', '标的价格', '标的价格变化率', '期权价格变化率', 'iv', 'Delta', 'Delta Cash', 'Gamma',
                 'Gamma Cash',
                 'Vega',
                 'Vega Cash', 'Theta', 'Theta Cash', 'rho', 'rho Cash'])
    i = 1
    while i <= 20:
        change = f"标的价格减小 {i}/100"
        S_change_prc = -i / 100
        S_c = S0 * (100 - i) / 100
        op_value_c = futOptPrice(S_c, K, T, r, old_iv, option_type)
        greeks_c = Greeks(S_c, K, T, r, old_iv, option_type)
        price_change_prc = (op_value_c - op_value_market) / op_value_market
        IF_c = pd.DataFrame(
            [[change, op_value_market, op_value_c, S_c, S_change_prc, price_change_prc, old_iv, greeks_c.Delta(),
              greeks_c.Delta() * unit,
              greeks_c.Gamma(),
              greeks_c.Gamma() * unit, greeks_c.Vega(), greeks_c.Vega() * unit,
              greeks_c.Theta(), greeks_c.Theta() * unit, greeks_c.rho(), greeks_c.rho() * unit]],
            columns=['标的价格变化', '当前期权价格', '期权价格', '标的价格', '标的价格变化率', '期权价格变化率', 'iv', 'Delta', 'Delta Cash', 'Gamma',
                     'Gamma Cash',
                     'Vega',
                     'Vega Cash', 'Theta', 'Theta Cash', 'rho', 'rho Cash'])
        PM_IV_information = pd.concat([IF_c, PM_IV_information], axis=0)
        i = i + 1

    k = 1
    while k <= 20:
        change = f"标的价格增加 {k}/1000"
        S_change_prc = k / 100
        S_c = S0 * (100 + k) / 100
        op_value_c = futOptPrice(S_c, K, T, r, old_iv, option_type)
        greeks_c = Greeks(S_c, K, T, r, old_iv, option_type)
        price_change_prc = (op_value_c - op_value_market) / op_value_market
        IF_c = pd.DataFrame(
            [[change, op_value_market, op_value_c, S_c, S_change_prc, price_change_prc, old_iv, greeks_c.Delta(),
              greeks_c.Delta() * unit,
              greeks_c.Gamma(),
              greeks_c.Gamma() * unit, greeks_c.Vega(), greeks_c.Vega() * unit,
              greeks_c.Theta(), greeks_c.Theta() * unit, greeks_c.rho(), greeks_c.rho() * unit]],
            columns=['标的价格变化', '当前期权价格', '期权价格', '标的价格', '标的价格变化率', '期权价格变化率', 'iv', 'Delta', 'Delta Cash', 'Gamma',
                     'Gamma Cash',
                     'Vega',
                     'Vega Cash', 'Theta', 'Theta Cash', 'rho', 'rho Cash'])
        PM_IV_information = pd.concat([PM_IV_information, IF_c], axis=0)
        k = k + 1

    if long_short == 'long':
        PM_IV_information = PM_IV_information
    elif long_short == 'short':
        PM_IV_information['当前期权价格'] = -PM_IV_information['当前期权价格']
        PM_IV_information['期权价格'] = -PM_IV_information['期权价格']
        PM_IV_information['Delta'] = -PM_IV_information['Delta']
        PM_IV_information['Delta Cash'] = -PM_IV_information['Delta Cash']
        PM_IV_information['Gamma'] = -PM_IV_information['Gamma']
        PM_IV_information['Gamma Cash'] = -PM_IV_information['Gamma Cash']
        PM_IV_information['Vega'] = -PM_IV_information['Vega']
        PM_IV_information['Vega Cash'] = -PM_IV_information['Vega Cash']
        PM_IV_information['Theta'] = -PM_IV_information['Theta']
        PM_IV_information['Theta Cash'] = -PM_IV_information['Theta Cash']
        PM_IV_information['rho'] = -PM_IV_information['rho']
        PM_IV_information['rho Cash'] = -PM_IV_information['rho Cash']

    return PM_IV_information.dropna()


# 时间流逝
def Time_variation(S0, K, T, r, op_value_market, option_type, long_short):
    unit = 1000
    T = T / 365
    old_iv = impv(S0, K, T, r, op_value_market, option_type)
    old_greeks = Greeks(S0, K, T, r, old_iv, option_type)
    PM_IV_information = pd.DataFrame(
        [["0", op_value_market, op_value_market, S0, 0, 0, old_iv, old_greeks.Delta(), old_greeks.Delta() * unit,
          old_greeks.Gamma(), old_greeks.Gamma() * unit, old_greeks.Vega(), old_greeks.Vega() * unit,
          old_greeks.Theta(), old_greeks.Theta() * unit, old_greeks.rho(), old_greeks.rho() * unit]],
        columns=['时间变化', '当前期权价格', '期权价格', '标的价格', '时间变化天数', '期权价格变化率', 'iv', 'Delta', 'Delta Cash', 'Gamma',
                 'Gamma Cash',
                 'Vega',
                 'Vega Cash', 'Theta', 'Theta Cash', 'rho', 'rho Cash'])
    i = 1
    while i < T * 365:
        change = f"时间流逝 {i}天"
        time_flow = i
        T_c = (T * 365 - time_flow) / 365
        op_value_c = futOptPrice(S0, K, T_c, r, old_iv, option_type)
        greeks_c = Greeks(S0, K, T_c, r, old_iv, option_type)
        price_change_prc = (op_value_c - op_value_market) / op_value_market
        IF_c = pd.DataFrame(
            [[change, op_value_market, op_value_c, S0, time_flow, price_change_prc, old_iv, greeks_c.Delta(),
              greeks_c.Delta() * unit,
              greeks_c.Gamma(),
              greeks_c.Gamma() * unit, greeks_c.Vega(), greeks_c.Vega() * unit,
              greeks_c.Theta(), greeks_c.Theta() * unit, greeks_c.rho(), greeks_c.rho() * unit]],
            columns=['时间变化', '当前期权价格', '期权价格', '标的价格', '时间变化天数', '期权价格变化率', 'iv', 'Delta', 'Delta Cash', 'Gamma',
                     'Gamma Cash',
                     'Vega',
                     'Vega Cash', 'Theta', 'Theta Cash', 'rho', 'rho Cash'])
        PM_IV_information = pd.concat([IF_c, PM_IV_information], axis=0)
        i = i + 1

    if long_short == 'long':
        PM_IV_information = PM_IV_information
    elif long_short == 'short':
        PM_IV_information['当前期权价格'] = -PM_IV_information['当前期权价格']
        PM_IV_information['期权价格'] = -PM_IV_information['期权价格']
        PM_IV_information['Delta'] = -PM_IV_information['Delta']
        PM_IV_information['Delta Cash'] = -PM_IV_information['Delta Cash']
        PM_IV_information['Gamma'] = -PM_IV_information['Gamma']
        PM_IV_information['Gamma Cash'] = -PM_IV_information['Gamma Cash']
        PM_IV_information['Vega'] = -PM_IV_information['Vega']
        PM_IV_information['Vega Cash'] = -PM_IV_information['Vega Cash']
        PM_IV_information['Theta'] = -PM_IV_information['Theta']
        PM_IV_information['Theta Cash'] = -PM_IV_information['Theta Cash']
        PM_IV_information['rho'] = -PM_IV_information['rho']
        PM_IV_information['rho Cash'] = -PM_IV_information['rho Cash']
    return PM_IV_information.dropna()


# 三维分析
def price_time_variation(S0, K, T, r, op_value_market, option_type, long_short):
    unit = 1000
    T = T / 365
    old_iv = impv(S0, K, T, r, op_value_market, option_type)

    old_greeks = Greeks(S0, K, T, r, old_iv, option_type)
    PM_IV_information = pd.DataFrame(
        [["0", 0, op_value_market, op_value_market, S0, 0, 0, 0, old_iv, old_greeks.Delta(), old_greeks.Delta() * unit,
          old_greeks.Gamma(), old_greeks.Gamma() * unit, old_greeks.Vega(), old_greeks.Vega() * unit,
          old_greeks.Theta(), old_greeks.Theta() * unit, old_greeks.rho(), old_greeks.rho() * unit]],
        columns=['时间变化', '标的价格变化', '当前期权价格', '期权价格', '标的价格', '时间变化天数', '标的价格变化率', '期权价格变化率', 'iv', 'Delta',
                 'Delta Cash', 'Gamma',
                 'Gamma Cash', 'Vega', 'Vega Cash',
                 'Theta', 'Theta Cash', 'rho', 'rho Cash'])
    i = T * 365
    while i >= 0:
        time_change = f"时间流逝 {i}天"
        time_flow = i
        T_c = (T * 365 - time_flow) / 365
        x = 30
        while x >= 0:
            target_change = f"标的价格减小 {x}/100"
            S_change_prc = -x / 100
            S_c = S0 * (100 - x) / 100
            op_value_c = futOptPrice(S_c, K, T_c, r, old_iv, option_type)
            c_iv = impv(S0, K, T, r, op_value_c, option_type)

            greeks_c = Greeks(S_c, K, T_c, r, old_iv, option_type)
            price_change_prc = (op_value_c - op_value_market) / op_value_market
            IF_c = pd.DataFrame([[time_change, target_change, op_value_market, op_value_c, S_c, time_flow, S_change_prc,
                                  price_change_prc, c_iv, greeks_c.Delta(), greeks_c.Delta() * unit, greeks_c.Gamma(),
                                  greeks_c.Gamma() * unit, greeks_c.Vega(), greeks_c.Vega() * unit,
                                  greeks_c.Theta(), greeks_c.Theta() * unit, greeks_c.rho(), greeks_c.rho() * unit]],
                                columns=['时间变化', '标的价格变化', '当前期权价格', '期权价格', '标的价格', '时间变化天数', '标的价格变化率', '期权价格变化率',
                                         'iv',
                                         'Delta', 'Delta Cash', 'Gamma', 'Gamma Cash', 'Vega', 'Vega Cash',
                                         'Theta', 'Theta Cash', 'rho', 'rho Cash'])
            PM_IV_information = pd.concat([IF_c, PM_IV_information], axis=0)
            x = x - 1

        y = 1
        while y <= 20:
            target_change = f"标的价格增加 {y}/1000"
            S_change_prc = y / 100
            S_c = S0 * (100 + y) / 100
            op_value_c = futOptPrice(S_c, K, T_c, r, old_iv, option_type)
            c_iv = impv(S0, K, T, r, op_value_c, option_type)
            greeks_c = Greeks(S_c, K, T_c, r, old_iv, option_type)
            price_change_prc = (op_value_c - op_value_market) / op_value_market
            IF_c = pd.DataFrame([[time_change, target_change, op_value_market, op_value_c, S_c, time_flow, S_change_prc,
                                  price_change_prc, c_iv, greeks_c.Delta(), greeks_c.Delta() * unit, greeks_c.Gamma(),
                                  greeks_c.Gamma() * unit, greeks_c.Vega(), greeks_c.Vega() * unit,
                                  greeks_c.Theta(), greeks_c.Theta() * unit, greeks_c.rho(), greeks_c.rho() * unit]],
                                columns=['时间变化', '标的价格变化', '当前期权价格', '期权价格', '标的价格', '时间变化天数', '标的价格变化率', '期权价格变化率',
                                         'iv',
                                         'Delta', 'Delta Cash', 'Gamma', 'Gamma Cash', 'Vega', 'Vega Cash',
                                         'Theta', 'Theta Cash', 'rho', 'rho Cash'])
            PM_IV_information = pd.concat([IF_c, PM_IV_information], axis=0)
            y = y + 1
        i -= 1

    if long_short == 'long':
        PM_IV_information = PM_IV_information
    elif long_short == 'short':
        PM_IV_information['当前期权价格'] = -PM_IV_information['当前期权价格']
        PM_IV_information['期权价格'] = -PM_IV_information['期权价格']
        PM_IV_information['Delta'] = -PM_IV_information['Delta']
        PM_IV_information['Delta Cash'] = -PM_IV_information['Delta Cash']
        PM_IV_information['Gamma'] = -PM_IV_information['Gamma']
        PM_IV_information['Gamma Cash'] = -PM_IV_information['Gamma Cash']
        PM_IV_information['Vega'] = -PM_IV_information['Vega']
        PM_IV_information['Vega Cash'] = -PM_IV_information['Vega Cash']
        PM_IV_information['Theta'] = -PM_IV_information['Theta']
        PM_IV_information['Theta Cash'] = -PM_IV_information['Theta Cash']
        PM_IV_information['rho'] = -PM_IV_information['rho']
        PM_IV_information['rho Cash'] = -PM_IV_information['rho Cash']

    return PM_IV_information.dropna()


def sigma_price_time_variation(S0, K, T, r, op_value_market, option_type, long_short):
    unit = 1000
    T = T / 365
    old_iv = impv(S0, K, T, r, op_value_market, option_type)
    old_greeks = Greeks(S0, K, T, r, old_iv, option_type)
    PM_IV_information = pd.DataFrame(
        [["0", 0, op_value_market, op_value_market, S0, 0, 0, 0, old_iv, old_greeks.Delta(), old_greeks.Delta() * unit,
          old_greeks.Gamma(), old_greeks.Gamma() * unit, old_greeks.Vega(), old_greeks.Vega() * unit,
          old_greeks.Theta(), old_greeks.Theta() * unit, old_greeks.rho(), old_greeks.rho() * unit]],
        columns=['时间变化', '标的价格变化', '当前期权价格', '期权价格', '标的价格', '时间变化天数', '标的价格变化率', '期权价格变化率', 'iv', 'Delta',
                 'Delta Cash', 'Gamma',
                 'Gamma Cash', 'Vega', 'Vega Cash',
                 'Theta', 'Theta Cash', 'rho', 'rho Cash'])
    c_iv = 0.1
    while c_iv <= 1:
        i = T * 365 - 1
        while i >= 1:
            time_change = f"时间流逝{i - 1}天"
            time_flow = i - 1
            T_c = (T * 365 - time_flow) / 365
            x = 15
            while x >= 0:
                target_change = f"标的价格减小{x}/100"
                S_change_prc = - x / 100
                S_c = S0 * (100 - x) / 100
                op_value_c = futOptPrice(S_c, K, T_c, r, c_iv, option_type)
                greeks_c = Greeks(S_c, K, T_c, r, c_iv, option_type)
                price_change_prc = (op_value_c - op_value_market) / op_value_market
                IF_c = pd.DataFrame(
                    [[time_change, target_change, op_value_market, op_value_c, S_c, time_flow, S_change_prc,
                      price_change_prc, c_iv, greeks_c.Delta(), greeks_c.Delta() * unit,
                      greeks_c.Gamma(),
                      greeks_c.Gamma() * unit, greeks_c.Vega(), greeks_c.Vega() * unit,
                      greeks_c.Theta(), greeks_c.Theta() * unit, greeks_c.rho(),
                      greeks_c.rho() * unit]],
                    columns=['时间变化', '标的价格变化', '当前期权价格', '期权价格', '标的价格', '时间变化天数', '标的价格变化率', '期权价格变化率', 'iv',
                             'Delta', 'Delta Cash', 'Gamma', 'Gamma Cash', 'Vega', 'Vega Cash',
                             'Theta', 'Theta Cash', 'rho', 'rho Cash'])
                PM_IV_information = pd.concat([IF_c, PM_IV_information], axis=0)
                x = x - 1
            y = 0
            while y <= 20:
                target_change = f"标的价格增加 {y}/100"
                S_change_prc = y / 100
                S_c = S0 * (100 + y) / 100
                op_value_c = futOptPrice(S_c, K, T_c, r, c_iv, option_type)
                greeks_c = Greeks(S_c, K, T_c, r, c_iv, option_type)
                price_change_prc = (op_value_c - op_value_market) / op_value_market
                IF_c = pd.DataFrame(
                    [[time_change, target_change, op_value_market, op_value_c, S_c, time_flow, S_change_prc,
                      price_change_prc, c_iv, greeks_c.Delta(), greeks_c.Delta() * unit,
                      greeks_c.Gamma(),
                      greeks_c.Gamma() * unit, greeks_c.Vega(), greeks_c.Vega() * unit,
                      greeks_c.Theta(), greeks_c.Theta() * unit, greeks_c.rho(),
                      greeks_c.rho() * unit]],
                    columns=['时间变化', '标的价格变化', '当前期权价格', '期权价格', '标的价格', '时间变化天数', '标的价格变化率', '期权价格变化率', 'iv',
                             'Delta', 'Delta Cash', 'Gamma', 'Gamma Cash', 'Vega', 'Vega Cash',
                             'Theta', 'Theta Cash', 'rho', 'rho Cash'])
                PM_IV_information = pd.concat([IF_c, PM_IV_information], axis=0)
                y = y + 1
            i -= 1
        c_iv += 0.1

    if long_short == 'long':
        PM_IV_information = PM_IV_information
    elif long_short == 'short':
        PM_IV_information['当前期权价格'] = -PM_IV_information['当前期权价格']
        PM_IV_information['期权价格'] = -PM_IV_information['期权价格']
        PM_IV_information['Delta'] = -PM_IV_information['Delta']
        PM_IV_information['Delta Cash'] = -PM_IV_information['Delta Cash']
        PM_IV_information['Gamma'] = -PM_IV_information['Gamma']
        PM_IV_information['Gamma Cash'] = -PM_IV_information['Gamma Cash']
        PM_IV_information['Vega'] = -PM_IV_information['Vega']
        PM_IV_information['Vega Cash'] = -PM_IV_information['Vega Cash']
        PM_IV_information['Theta'] = -PM_IV_information['Theta']
        PM_IV_information['Theta Cash'] = -PM_IV_information['Theta Cash']
        PM_IV_information['rho'] = -PM_IV_information['rho']
        PM_IV_information['rho Cash'] = -PM_IV_information['rho Cash']

    return PM_IV_information.dropna()


class All_plot:
    def __init__(self, DATA, DATA_sigma, DATA_price, DATA_time):
        self.DATA = DATA
        self.DATA_sigma = DATA_sigma
        self.DATA_price = DATA_price
        self.DATA_time = DATA_time

    def plot_price(self):
        balance_data = self.DATA[abs(self.DATA['期权价格变化率']) < 0.05]
        ax1 = plt.axes(projection='3d')
        cm = plt.cm.get_cmap('RdYlBu')

        x = np.array(self.DATA['时间变化天数'])
        y = np.array(self.DATA['标的价格变化率'])
        z = np.array(self.DATA['iv'])
        c = np.array(self.DATA['期权价格变化率'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax1.yaxis.set_major_formatter(formatter)
        ax1.zaxis.set_major_formatter(formatter)
        ax1.scatter3D(x, y, z, c=c, marker='o', cmap=cm, s=10)
        sc = ax1.scatter3D(x, y, z, c=c, marker='o', cmap=cm, s=10)
        ax1.set_xlabel('时间变化天数')
        ax1.set_ylabel('标的价格变化')
        ax1.set_zlabel('隐含波动率')
        #ax1.zaxis.set_rotate_label(False)
        #ax1.set_zlabel('隐含波动率', rotation=90)
        ax1.set_title('期权价格变动')
        plt.colorbar(sc, label='期权价格变化率', pad=0.2, shrink=0.6, location='right')

        # ax1.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='隐含波动率')
        # ax1.set_title('Option price plot')
        # x2 = np.array(balance_data['时间变化天数'])
        # y2 = np.array(balance_data['标的价格变化率'])
        # z2 = np.array(balance_data['iv'])
        # ax1.scatter3D(x2, y2, z2, marker='o', color='black', s=20)
        plt.show()

    def plot_price_surface(self):
        balance_data = self.DATA[abs(self.DATA['期权价格变化率']) < 0.05]
        ax1 = plt.axes(projection='3d')
        cm = plt.cm.get_cmap('RdYlBu')

        x = np.array(self.DATA['时间变化天数'].values)
        y = np.array(self.DATA['标的价格变化率'].values)
        z = np.array(self.DATA['iv'].values)
        c = np.array(self.DATA['期权价格变化率'].values)

        X, Y = np.meshgrid(x, y)
        Z, _ = np.meshgrid(z, np.zeros_like(y))
        W = X+Y+Z

        ax1.plot_surface(X, Y, W, cmap='YlOrRd')
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax1.yaxis.set_major_formatter(formatter)
        ax1.zaxis.set_major_formatter(formatter)
        selected_indices = [60, 40]
        for i in selected_indices:
            ax1.text(x[i], y[i], Z[i, 0], np.round(c[i], 2), fontsize=10, color='black')
        ax1.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='隐含波动率')
        ax1.set_title('期权价格变动')
        #plt.colorbar(sc, label='期权价格变化率')

        # x2 = np.array(balance_data['时间变化天数'])
        # y2 = np.array(balance_data['标的价格变化率'])
        # z2 = np.array(balance_data['iv'])
        # X2, Y2 = np.meshgrid(x2, y2)
        # Z2, _ = np.meshgrid(z2, np.zeros_like(y2))
        # W2 = X2 + Y2 + Z2
        # ax1.plot_surface(X2, Y2, W2)
        plt.show()


    def plot_delta(self):
        ax2 = plt.axes(projection='3d')
        cm = plt.cm.get_cmap('RdYlBu')
        x = np.array(self.DATA['时间变化天数'])
        y = np.array(self.DATA['标的价格变化率'])
        z = np.array(self.DATA['iv'])
        c = np.array(self.DATA['Delta Cash'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax2.yaxis.set_major_formatter(formatter)
        ax2.scatter3D(x, y, z, c=c, cmap=cm, s=10)
        ax2.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='隐含波动率')
        ax2.set_title('Delta plot')
        plt.show()

    def plot_gamma(self):
        ax3 = plt.axes(projection='3d')
        x = np.array(self.DATA['时间变化天数'])
        y = np.array(self.DATA['标的价格变化率'])
        z = np.array(self.DATA['iv'])
        c = np.array(self.DATA['Gamma Cash'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax3.yaxis.set_major_formatter(formatter)
        ax3.scatter3D(x, y, z, c=c, cmap='summer', s=10)
        ax3.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='隐含波动率')
        ax3.set_title('Gamma plot')
        plt.show()

    def plot_vega(self):
        ax4 = plt.axes(projection='3d')
        x = np.array(self.DATA['时间变化天数'])
        y = np.array(self.DATA['标的价格变化率'])
        z = np.array(self.DATA['iv'])
        c = np.array(self.DATA['Vega Cash'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax4.yaxis.set_major_formatter(formatter)
        ax4.scatter3D(x, y, z, c=c, cmap='spring', s=10)
        ax4.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='隐含波动率')
        ax4.set_title('Vega plot')
        plt.show()

    def plot_theta(self):
        ax5 = plt.axes(projection='3d')
        x = np.array(self.DATA['时间变化天数'])
        y = np.array(self.DATA['标的价格变化率'])
        z = np.array(self.DATA['iv'])
        c = np.array(self.DATA['Theta'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax5.yaxis.set_major_formatter(formatter)
        ax5.zaxis.set_major_formatter(formatter)
        ax5.scatter3D(x, y, z, c=c, cmap='summer', s=10)
        ax5.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='隐含波动率')
        ax5.set_title('Theta plot')
        plt.show()

    def plot_rho(self):
        ax6 = plt.axes(projection='3d')
        x = np.array(self.DATA['时间变化天数'])
        y = np.array(self.DATA['标的价格变化率'])
        z = np.array(self.DATA['iv'])
        c = np.array(self.DATA['rho Cash'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax6.yaxis.set_major_formatter(formatter)
        ax6.scatter3D(x, y, z, c=c, cmap='cool', s=10)
        ax6.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='隐含波动率')
        ax6.set_title('rho plot')
        plt.show()

    def plot_2_sigma(self):
        x = self.DATA_sigma['iv']
        y = self.DATA_sigma['期权价格变化率']
        plt.scatter(x, y, c=y, cmap='plasma')

    def plot_2_price(self):
        x = self.DATA_price['标的价格变化率']
        y = self.DATA_price['期权价格变化率']
        plt.scatter(x, y, c=y, cmap='plasma')

    def plot_2_time(self):
        x = self.DATA_price['时间变化天数']
        y = self.DATA_price['期权价格变化率']
        plt.scatter(x, y, c=y, cmap='plasma')

    # 时间为变量， 观察该日头寸收益率随标的价格和波动率的走向
    def plot_set_time(self, set_time):
        set_time_data = self.DATA[self.DATA['时间变化天数'] == set_time]
        ax6 = plt.axes(projection='3d')
        x = np.array(set_time_data['iv'])
        y = np.array(set_time_data['标的价格变化率'])
        z = np.array(set_time_data['期权价格变化率'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax6.yaxis.set_major_formatter(formatter)
        ax6.zaxis.set_major_formatter(formatter)
        ax6.scatter3D(x, y, z, c=z, cmap='cool', s=10)
        ax6.set(xlabel='iv', ylabel='标的价格变化率', zlabel='期权价格变化率')
        ax6.set_title(f'第{set_time}天头寸价值变化率与（波动率&标的价格变化率）的关系')
        plt.show()

    def plot_out(self):
        fig = plt.figure(figsize=(12, 6), facecolor='w')
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)
        cm = plt.cm.get_cmap('RdYlBu')
        ax1 = fig.add_subplot(331, projection='3d')
        x = np.array(self.DATA['时间变化天数'])
        y = np.array(self.DATA['标的价格变化率'])
        z = np.array(self.DATA['期权价格变化率'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax1.yaxis.set_major_formatter(formatter)
        ax1.zaxis.set_major_formatter(formatter)
        ax1.scatter3D(x, y, z, c=z, marker='o', cmap=cm)
        ax1.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='期权价格变化率')
        ax1.set_title('Option price plot')

        ax2 = fig.add_subplot(332, projection='3d')
        x = np.array(self.DATA['时间变化天数'])
        y = np.array(self.DATA['标的价格变化率'])
        z = np.array(self.DATA['Delta Cash'])
        formatter = ticker.FormatStrFormatter('%1.1f')
        ax2.yaxis.set_major_formatter(formatter)
        ax2.zaxis.set_major_formatter(formatter)
        ax2.scatter3D(x, y, z, c=z, cmap='Greens', s=7)
        ax2.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='Delta Cash')
        ax2.set_title('Delta plot')

        ax3 = fig.add_subplot(333, projection='3d')
        x = np.array(self.DATA['时间变化天数'])
        y = np.array(self.DATA['标的价格变化率'])
        z = np.array(self.DATA['Gamma Cash'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax3.yaxis.set_major_formatter(formatter)
        ax3.zaxis.set_major_formatter(formatter)
        ax3.scatter3D(x, y, z, c=z, cmap='plasma', s=7)
        ax3.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='Gamma Cash')
        ax3.set_title('Gamma plot')

        ax4 = fig.add_subplot(334, projection='3d')
        x = np.array(self.DATA['时间变化天数'])
        y = np.array(self.DATA['标的价格变化率'])
        z = np.array(self.DATA['Vega Cash'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax4.yaxis.set_major_formatter(formatter)
        ax4.zaxis.set_major_formatter(formatter)
        ax4.scatter3D(x, y, z, c=z, cmap='summer', s=7)
        ax4.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='Vega Cash')
        ax4.set_title('Vega plot')

        ax5 = fig.add_subplot(335, projection='3d')
        x = np.array(self.DATA['时间变化天数'])
        y = np.array(self.DATA['标的价格变化率'])
        z = np.array(self.DATA['Theta Cash'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax5.yaxis.set_major_formatter(formatter)
        ax5.zaxis.set_major_formatter(formatter)
        ax5.scatter3D(x, y, z, c=z, cmap='autumn', s=7)
        ax5.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='Theta Cash')
        ax5.set_title('Theta plot')

        ax6 = fig.add_subplot(336, projection='3d')
        x = np.array(self.DATA['时间变化天数'])
        y = np.array(self.DATA['标的价格变化率'])
        z = np.array(self.DATA['rho Cash'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax6.yaxis.set_major_formatter(formatter)
        ax6.zaxis.set_major_formatter(formatter)
        ax6.scatter3D(x, y, z, c=z, cmap='winter', s=7)
        ax6.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='rho Cash')
        ax6.set_title('rho plot')

        ax7 = fig.add_subplot(337)
        x = np.array(self.DATA_sigma['iv'])
        y = np.array(self.DATA_sigma['期权价格变化率'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax7.xaxis.set_major_formatter(formatter)
        ax7.yaxis.set_major_formatter(formatter)
        ax7.scatter(x, y, c=y, cmap='plasma', s=7)
        ax7.set(xlabel='波动率', ylabel='期权价格变化率')
        ax7.set_title('2D_波动率')
        ax7.grid(linestyle='-.')

        ax8 = fig.add_subplot(338)
        x = np.array(self.DATA_price['标的价格变化率'])
        y = np.array(self.DATA_price['期权价格变化率'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax8.xaxis.set_major_formatter(formatter)
        ax8.yaxis.set_major_formatter(formatter)
        ax8.scatter(x, y, c=y, cmap='plasma', s=7)
        ax8.set(xlabel='标的价格变化率', ylabel='期权价格变化率')
        ax8.set_title('2D_标的价格')
        ax8.grid(linestyle='-.')

        ax9 = fig.add_subplot(339)
        x = np.array(self.DATA_time['时间变化天数'])
        y = np.array(self.DATA_time['期权价格变化率'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax9.xaxis.set_major_formatter(formatter)
        ax9.yaxis.set_major_formatter(formatter)
        ax9.scatter(x, y, c=y, cmap='plasma', s=7)
        ax9.set(xlabel='时间变化天数', ylabel='期权价格变化率')
        ax9.set_title('2D_时间天数')
        ax9.grid(linestyle='-.')

        plt.show()

    def plot_out_2(self):
        fig = plt.figure(figsize=(12, 6), facecolor='w')
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)
        cm = plt.cm.get_cmap('RdYlBu')
        ax1 = fig.add_subplot(331, projection='3d')
        x = np.array(self.DATA['时间变化天数'])
        y = np.array(self.DATA['标的价格变化率'])
        z = np.array(self.DATA['iv'])
        c = np.array(self.DATA['期权价格变化率'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax1.yaxis.set_major_formatter(formatter)
        ax1.zaxis.set_major_formatter(formatter)
        ax1.scatter3D(x, y, z, c=c, marker='o', cmap=cm)
        ax1.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='隐含波动率')
        ax1.set_title('Option price plot')

        ax2 = fig.add_subplot(332, projection='3d')
        x = np.array(self.DATA['时间变化天数'])
        y = np.array(self.DATA['标的价格变化率'])
        z = np.array(self.DATA['iv'])
        c = np.array(self.DATA['Delta Cash'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax2.yaxis.set_major_formatter(formatter)
        ax2.scatter3D(x, y, z, c=c, cmap=cm, s=10)
        ax2.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='隐含波动率')
        ax2.set_title('Delta plot')

        ax3 = fig.add_subplot(333, projection='3d')
        x = np.array(self.DATA['时间变化天数'])
        y = np.array(self.DATA['标的价格变化率'])
        z = np.array(self.DATA['iv'])
        c = np.array(self.DATA['Gamma Cash'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax3.yaxis.set_major_formatter(formatter)
        ax3.scatter3D(x, y, z, c=c, cmap=cm, s=10)
        ax3.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='隐含波动率')
        ax3.set_title('Gamma plot')

        ax4 = fig.add_subplot(334, projection='3d')
        x = np.array(self.DATA['时间变化天数'])
        y = np.array(self.DATA['标的价格变化率'])
        z = np.array(self.DATA['iv'])
        c = np.array(self.DATA['Vega Cash'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax4.yaxis.set_major_formatter(formatter)
        ax4.scatter3D(x, y, z, c=c, cmap=cm, s=10)
        ax4.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='隐含波动率')
        ax4.set_title('Vega plot')

        ax5 = fig.add_subplot(335, projection='3d')
        x = np.array(self.DATA['时间变化天数'])
        y = np.array(self.DATA['标的价格变化率'])
        z = np.array(self.DATA['iv'])
        c = np.array(self.DATA['Theta Cash'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax5.yaxis.set_major_formatter(formatter)
        ax5.zaxis.set_major_formatter(formatter)
        ax5.scatter3D(x, y, z, c=c, cmap=cm, s=10)
        ax5.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='隐含波动率')
        ax5.set_title('Theta plot')

        ax6 = fig.add_subplot(336, projection='3d')
        x = np.array(self.DATA['时间变化天数'])
        y = np.array(self.DATA['标的价格变化率'])
        z = np.array(self.DATA['iv'])
        c = np.array(self.DATA['rho Cash'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax6.yaxis.set_major_formatter(formatter)
        ax6.scatter3D(x, y, z, c=c, cmap=cm, s=10)
        ax6.set(xlabel='时间变化天数', ylabel='标的价格变化', zlabel='隐含波动率')
        ax6.set_title('rho plot')

        ax7 = fig.add_subplot(337)
        x = np.array(self.DATA_sigma['iv'])
        y = np.array(self.DATA_sigma['期权价格变化率'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax7.xaxis.set_major_formatter(formatter)
        ax7.yaxis.set_major_formatter(formatter)
        ax7.scatter(x, y, c=y, cmap=cm, s=10)
        ax7.set(xlabel='波动率', ylabel='期权价格变化率')
        ax7.set_title('2D_波动率')
        ax7.grid(linestyle='-.')

        ax8 = fig.add_subplot(338)
        x = np.array(self.DATA_price['标的价格变化率'])
        y = np.array(self.DATA_price['期权价格变化率'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax8.xaxis.set_major_formatter(formatter)
        ax8.yaxis.set_major_formatter(formatter)
        ax8.scatter(x, y, c=y, cmap=cm, s=10)
        ax8.set(xlabel='标的价格变化率', ylabel='期权价格变化率')
        ax8.set_title('2D_标的价格')
        ax8.grid(linestyle='-.')

        ax9 = fig.add_subplot(339)
        x = np.array(self.DATA_time['时间变化天数'])
        y = np.array(self.DATA_time['期权价格变化率'])
        formatter = ticker.FormatStrFormatter('%1.2f')
        ax9.xaxis.set_major_formatter(formatter)
        ax9.yaxis.set_major_formatter(formatter)
        ax9.scatter(x, y, c=y, cmap=cm, s=10)
        ax9.set(xlabel='时间变化天数', ylabel='期权价格变化率')
        ax9.set_title('2D_时间天数')
        ax9.grid(linestyle='-.')

        plt.show()


def combine_portfolio(DATA1, DATA2):
    combine_data = pd.DataFrame()
    merge_data = pd.merge(DATA1, DATA2, on=['时间变化', '标的价格变化', '标的价格', '时间变化天数', '标的价格变化率', 'iv'], how='inner')
    combine_data[['时间变化', '标的价格变化', '标的价格', '时间变化天数', '标的价格变化率', 'iv']] = merge_data[
        ['时间变化', '标的价格变化', '标的价格', '时间变化天数', '标的价格变化率', 'iv']]
    combine_data['期权价格'] = merge_data['期权价格_x'] + merge_data['期权价格_y']
    combine_data['当前期权价格'] = merge_data['当前期权价格_x'] + merge_data['当前期权价格_y']
    combine_data['期权价格变化率'] = (combine_data['期权价格'] - combine_data['当前期权价格']) / combine_data['当前期权价格']
    combine_data['Delta'] = merge_data['Delta_x'] + merge_data['Delta_y']
    combine_data['Delta Cash'] = merge_data['Delta Cash_x'] + merge_data['Delta Cash_y']
    combine_data['Gamma'] = merge_data['Gamma_x'] + merge_data['Gamma_y']
    combine_data['Gamma Cash'] = merge_data['Gamma Cash_x'] + merge_data['Gamma Cash_y']
    combine_data['Vega'] = merge_data['Vega_x'] + merge_data['Vega_y']
    combine_data['Vega Cash'] = merge_data['Vega Cash_x'] + merge_data['Vega Cash_y']
    combine_data['Theta'] = merge_data['Theta_x'] + merge_data['Theta_y']
    combine_data['Theta Cash'] = merge_data['Theta Cash_x'] + merge_data['Theta Cash_y']
    combine_data['rho'] = merge_data['rho_x'] + merge_data['rho_y']
    combine_data['rho Cash'] = merge_data['rho Cash_x'] + merge_data['rho Cash_y']
    return combine_data


def combine_iv_inf(DATA1, DATA2):
    combine_data = pd.DataFrame()
    merge_data = pd.merge(DATA1, DATA2, on=['IV变化', '标的价格', 'iv'], how='inner')
    combine_data[['IV变化', '标的价格', 'iv']] = merge_data[['IV变化', '标的价格', 'iv']]
    combine_data['期权价格'] = merge_data['期权价格_x'] + merge_data['期权价格_y']
    combine_data['当前期权价格'] = merge_data['当前期权价格_x'] + merge_data['当前期权价格_y']
    combine_data['期权价格变化率'] = (combine_data['期权价格'] - combine_data['当前期权价格']) / combine_data['当前期权价格']
    combine_data['Delta'] = merge_data['Delta_x'] + merge_data['Delta_y']
    combine_data['Delta Cash'] = merge_data['Delta Cash_x'] + merge_data['Delta Cash_y']
    combine_data['Gamma'] = merge_data['Gamma_x'] + merge_data['Gamma_y']
    combine_data['Gamma Cash'] = merge_data['Gamma Cash_x'] + merge_data['Gamma Cash_y']
    combine_data['Vega'] = merge_data['Vega_x'] + merge_data['Vega_y']
    combine_data['Vega Cash'] = merge_data['Vega Cash_x'] + merge_data['Vega Cash_y']
    combine_data['Theta'] = merge_data['Theta_x'] + merge_data['Theta_y']
    combine_data['Theta Cash'] = merge_data['Theta Cash_x'] + merge_data['Theta Cash_y']
    combine_data['rho'] = merge_data['rho_x'] + merge_data['rho_y']
    combine_data['rho Cash'] = merge_data['rho Cash_x'] + merge_data['rho Cash_y']
    return combine_data


def combine_target_price_inf(DATA1, DATA2):
    combine_data = pd.DataFrame()
    merge_data = pd.merge(DATA1, DATA2, on=['标的价格变化', '标的价格', '标的价格变化率'], how='inner')
    combine_data[['标的价格变化', '标的价格', '标的价格变化率']] = merge_data[['标的价格变化', '标的价格', '标的价格变化率']]
    combine_data['期权价格'] = merge_data['期权价格_x'] + merge_data['期权价格_y']
    combine_data['当前期权价格'] = merge_data['当前期权价格_x'] + merge_data['当前期权价格_y']
    combine_data['期权价格变化率'] = (combine_data['期权价格'] - combine_data['当前期权价格']) / combine_data['当前期权价格']
    combine_data['Delta'] = merge_data['Delta_x'] + merge_data['Delta_y']
    combine_data['Delta Cash'] = merge_data['Delta Cash_x'] + merge_data['Delta Cash_y']
    combine_data['Gamma'] = merge_data['Gamma_x'] + merge_data['Gamma_y']
    combine_data['Gamma Cash'] = merge_data['Gamma Cash_x'] + merge_data['Gamma Cash_y']
    combine_data['Vega'] = merge_data['Vega_x'] + merge_data['Vega_y']
    combine_data['Vega Cash'] = merge_data['Vega Cash_x'] + merge_data['Vega Cash_y']
    combine_data['Theta'] = merge_data['Theta_x'] + merge_data['Theta_y']
    combine_data['Theta Cash'] = merge_data['Theta Cash_x'] + merge_data['Theta Cash_y']
    combine_data['rho'] = merge_data['rho_x'] + merge_data['rho_y']
    combine_data['rho Cash'] = merge_data['rho Cash_x'] + merge_data['rho Cash_y']
    return combine_data


def combine_time_inf(DATA1, DATA2):
    combine_data = pd.DataFrame()
    merge_data = pd.merge(DATA1, DATA2, on=['时间变化', '时间变化天数', '标的价格'], how='inner')
    combine_data[['时间变化', '时间变化天数', '标的价格']] = merge_data[['时间变化', '时间变化天数', '标的价格']]
    combine_data['期权价格'] = merge_data['期权价格_x'] + merge_data['期权价格_y']
    combine_data['当前期权价格'] = merge_data['当前期权价格_x'] + merge_data['当前期权价格_y']
    combine_data['期权价格变化率'] = (combine_data['期权价格'] - combine_data['当前期权价格']) / combine_data['当前期权价格']
    combine_data['Delta'] = merge_data['Delta_x'] + merge_data['Delta_y']
    combine_data['Delta Cash'] = merge_data['Delta Cash_x'] + merge_data['Delta Cash_y']
    combine_data['Gamma'] = merge_data['Gamma_x'] + merge_data['Gamma_y']
    combine_data['Gamma Cash'] = merge_data['Gamma Cash_x'] + merge_data['Gamma Cash_y']
    combine_data['Vega'] = merge_data['Vega_x'] + merge_data['Vega_y']
    combine_data['Vega Cash'] = merge_data['Vega Cash_x'] + merge_data['Vega Cash_y']
    combine_data['Theta'] = merge_data['Theta_x'] + merge_data['Theta_y']
    combine_data['Theta Cash'] = merge_data['Theta Cash_x'] + merge_data['Theta Cash_y']
    combine_data['rho'] = merge_data['rho_x'] + merge_data['rho_y']
    combine_data['rho Cash'] = merge_data['rho Cash_x'] + merge_data['rho Cash_y']
    return combine_data


if __name__ == '__main__':
    # 构造圣诞树形期权
    # 圣诞树形期权多头
    # Test
    b1 = sigma_price_time_variation(680, 650, 17, rf, 42.1, 'C', 'long')
    b2 = sigma_price_time_variation(680, 690, 17, rf, 18.2, 'P', 'short')
    b3 = sigma_price_time_variation(680, 720, 17, rf, 8.6, 'C', 'short')
    o1 = combine_portfolio(b1, b2)
    o1 = combine_portfolio(o1, b3)

    c1 = IV_variation(680, 650, 17, rf, 42.1, 'C', 'long')
    c2 = IV_variation(680, 690, 17, rf, 18.2, 'P', 'short')
    c3 = IV_variation(680, 720, 17, rf, 8.6, 'C', 'short')
    o2 = combine_iv_inf(c1, c2)
    o2 = combine_iv_inf(o2, c3)

    d1 = Target_price_variation(680, 650, 17, rf, 42.1, 'C', 'long')
    d2 = Target_price_variation(680, 690, 17, rf, 18.2, 'P', 'short')
    d3 = Target_price_variation(680, 720, 17, rf, 8.6, 'C', 'short')
    o3 = combine_target_price_inf(d1, d2)
    o3 = combine_target_price_inf(o3, d3)

    e1 = Time_variation(680, 650, 17, rf, 42.1, 'C', 'long')
    e2 = Time_variation(680, 690, 17, rf, 18.2, 'P', 'short')
    e3 = Time_variation(680, 720, 17, rf, 8.6, 'C', 'short')
    o4 = combine_time_inf(e1, e2)
    o4 = combine_time_inf(o4, e3)

    pl = All_plot(o1, o2, o3, o4)

    pl.plot_price()
    pl.plot_out_2()
    pl.plot_set_time(5)
    pl.plot_price_surface()


