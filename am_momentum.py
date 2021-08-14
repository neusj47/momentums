# 절대 모멘텀 전략
# 0. Input parameter 입력
# 1. 데이터 가져오기
# 2. 절대모멘텀 trading signal 생성
# 3. 수익률 계산
# 4. 결과비교

import pandas as pd
import numpy as np
import datetime
import pandas_datareader as pdr


# 0. parameter 입력
TICKER = '^KS11'
start_date = '2000-01-01'
end_date = datetime.datetime.today()
cost = 0
lookback_m1 = 1
lookback_m2 = 12

# 1. 데이터 가져오기
df = pdr.get_data_yahoo(TICKER, start = start_date, end = end_date)
price_df = df.loc[:,['Adj Close']].copy()
price_df = price_df.rename_axis('Date').reset_index()
price_df['Date'] = pd.to_datetime(price_df['Date'])
# book = price_df.copy()

# 2. trading signal 생성
def get_am_signal(price_df, lookback_m1, lookback_m2) :
    month_list = price_df['Date'].map(lambda x : datetime.datetime.strftime(x, '%Y-%m')).unique()
    rebal_date= pd.DataFrame()
    for m in month_list:
        rebal_date = rebal_date.append(price_df[price_df['Date'].map(lambda x : datetime.datetime.strftime(x, '%Y-%m'))== m].iloc[-1])
    rebal_date.set_index(['Date'],inplace=True)
    rebal_date['Rct_Adj Close'] = rebal_date.shift(lookback_m1)['Adj Close']
    rebal_date['Pst_Adj Close'] = rebal_date.shift(lookback_m2)['Adj Close']
    rebal_date = rebal_date.dropna(axis =0)
    rebal_date['signal'] = np.where(rebal_date['Rct_Adj Close'] / rebal_date['Pst_Adj Close'] > 1,1,0)
    rebal_date['YYYY-MM'] = rebal_date.index.map(lambda x : datetime.datetime.strftime(x, '%Y-%m'))
    price_df['YYYY-MM'] = price_df['Date'].map(lambda x : datetime.datetime.strftime(x, '%Y-%m'))
    book = pd.merge(price_df, rebal_date[['Rct_Adj Close', 'Pst_Adj Close', 'signal', 'YYYY-MM']], how = 'left', on = 'YYYY-MM')
    book = book.fillna(0)
    return book

book = get_am_signal(price_df, lookback_m1, lookback_m2)

# 2. 수익률 계산
def get_return(book) :
    for i in book.index:
        if book.loc[i, 'signal'] == 1 and book.shift(1).loc[i, 'signal'] == 0:
            buy = book.loc[i, 'Adj Close']
        elif book.loc[i, 'signal'] == 1 and book.shift(1).loc[i, 'signal'] == 1:
            current = book.loc[i, 'Adj Close']
            rtn = ((current - buy) / buy) + 1
            book.loc[i, 'return'] = rtn
        elif book.loc[i, 'signal'] == 0 and book.shift(1).loc[i, 'signal'] == 1:
            sell = book.loc[i, 'Adj Close']
            rtn = ((sell - buy) / buy) + 1
            book.loc[i, 'return'] = rtn
    acc_rtn = 1.0
    for i in book.index:
        if book.loc[i, 'signal'] == 0 and book.shift(1).loc[i, 'signal'] == 1:
            rtn = book.loc[i, 'return']
            acc_rtn = acc_rtn * rtn
            book.loc[i:, 'acc return'] = acc_rtn
        book['return_BM'] = book['Adj Close'].pct_change().fillna(0)
        book['acc return_BM'] = ((1 + book['Adj Close'].pct_change()).cumprod()-1).fillna(0)
    book['return'] = book['return'] - 1
    book['acc return'] = book['acc return'] - 1
    result = book[['Date','Adj Close','return','acc return','acc return_BM','signal']]
    return result

result = get_return(book)

# 결과
# import matplotlib.pyplot as plt
# fig, ax1 = plt.subplots()
# ax1.set_xlabel('StdDate')
# ax1.set_ylabel('Rtn(cum)')
# line1 = ax1.plot(result['Date'],result['acc return'],color='orangered', label ='AB_Momentum')
# line2 = ax1.plot(result['Date'],result['acc return_BM'],color='dodgerblue', label =TICKER)
# lines =line1 + line2
# labels = [l.get_label() for l in lines]
# ax1.legend(lines,labels)
#
# plt.figure(figsize=(15,7))
# plt.title('AM_signal')
# result['Adj Close'].plot(color = 'paleturquoise',linestyle='--')
# result['acc return'].plot(linestyle='--')
# result[result['signal']==1]['Adj Close'].plot(color='r', linestyle='None', marker='^')
# plt.legend()
# plt.show()

