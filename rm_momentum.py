# 상대모멘텀 전략
# 0. parameter 입력
# 1. 데이터 가져오기
# 2. trading signal 생성하기
# 3. 수익률 산출하기
# 4. 성과지표 생성

import pandas as pd
import numpy as np
import datetime
import pandas_datareader as pdr
import FinanceDataReader as fdr
import matplotlib.pyplot as plt

# 0. parameter 입력
TICKER = ['SPY', 'QQQ', 'EFA', 'IWD', 'IJH', 'IWM', 'XLK', 'VNQ']
selected_num = 3
lookback = 1
start_date = '2020-06-01'
end_date = datetime.datetime.today()

# 1. 데이터 가져오기
def get_data(TICKER, start_date, end_date) :
    '''
        Parameters
                ----------
                TICKER : list
                    Target TICKER list
                start_date : datetime
                    start_date
                end_date : datetime
                    end_date
        Returns
                -------
                df : dataframe
                    Historical daily Adj Close
    '''
    df = pd.DataFrame(pdr.get_data_yahoo(TICKER, start = start_date, end = end_date)['Adj Close'])
    df.columns = TICKER
    return df

df = get_data(TICKER, start_date, end_date)

# 2. trading signal 생성하기
def get_rm_signal(df, lookback) :
    '''
         Parameters
                 ----------
                 df : dataframe
                    Historical daily Adj Close
                 lookback : int
                     Lookback period for Calculating Momentum
         Returns
                 -------
                 signal : dataframe
                     Trading signal for Relative Momentum
     '''
    month_list = df.index.map(lambda x : datetime.datetime.strftime(x, '%Y-%m')).unique()
    rebal_date= pd.DataFrame()
    for m in month_list:
        rebal_date = rebal_date.append(df[df.index.map(lambda x : datetime.datetime.strftime(x, '%Y-%m')) == m].iloc[-1])
    rebal_date.columns = TICKER
    rebal_date = rebal_date/rebal_date.shift(lookback)
    signal = pd.DataFrame((rebal_date.rank(axis=1, ascending = False) <= selected_num).applymap(lambda x : '1' if x else '0'))
    signal = signal.shift(1).fillna(0)
    return signal

signal = get_rm_signal(df, lookback)

# 3. 수익률 산출하기

def get_rm_return(df,signal) :
    '''
         Parameters
                 ----------
                 df : dataframe
                    Historical daily Adj Close
                 signal : dataframe
                     Trading signal for Relative Momentum
         Returns
                 -------
                 result : dataframe
                     Relative Momentum portfolio's return
    '''
    df = df.rename_axis('Date').reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df['YYYY-MM'] = df['Date'].map(lambda x : datetime.datetime.strftime(x, '%Y-%m'))
    signal['YYYY-MM'] =signal.index.map(lambda x : datetime.datetime.strftime(x, '%Y-%m'))
    book = pd.merge(df[['Date','YYYY-MM']], signal, on = 'YYYY-MM', how = 'left')
    book.set_index(['Date'],inplace=True)
    book = book[TICKER].astype(float)
    df = df[['Date', 'SPY', 'QQQ', 'EFA', 'IWD','IJH','XLK','VNQ']]
    df.set_index(['Date'],inplace=True)
    df = df.pct_change().fillna(0)
    result = pd.DataFrame(((book * df) * 1 / selected_num).sum(axis=1))
    return result, book

result = get_rm_return(df,signal)[0]
book = get_rm_return(df,signal)[1] * 1/selected_num

# Cumulative Compounded Returns for RelativeMomentum
plt.figure(figsize=(17,7))
plt.title('Relative Momentum Return')
plt.plot((1 + result).cumprod() - 1, label = 'RM_Momentum')
plt.legend()
plt.show()

# Cross-Sectional Weights
plt.figure(figsize=(17,7))
plt.title('Cross-Sectional Weights')
plt.stackplot(book.index, np.transpose(book),labels = book.columns)
plt.legend()
plt.show()