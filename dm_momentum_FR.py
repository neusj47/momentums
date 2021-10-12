# 듀얼모멘텀 전략
# 0. parameter 입력
# 1. 데이터 가져오기
# 2. trading signal 생성하기
# 3. 수익률 산출하기
# 4. 성과지표 생성

import pandas as pd
import numpy as np
import datetime
import pandas_datareader as pdr
import matplotlib.pyplot as plt

# 0. parameter 입력
TICKER = ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']
selected_num = 1
lookback_m = 1
lookback_d = lookback_m * 30
start_date = '2015-09-30'
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
bm = pd.DataFrame(pdr.get_data_yahoo('SPY', start = start_date, end = end_date)['Adj Close'])

# 2. trading signal 생성하기
def get_rm_signal_m(df, lookback_m,selected_num) :
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
        try:
            rebal_date = rebal_date.append(
                df[df.index.map(lambda x: datetime.datetime.strftime(x, '%Y-%m')) == m].iloc[-1])
        except Exception as e:
            print("Error : ", str(e))
        pass
    rebal_date.columns = TICKER
    rebal_date = rebal_date/rebal_date.shift(lookback_m)
    recent_returns = df.pct_change(lookback_m*30)
    rebal_date.iloc[len(rebal_date) - 1] = recent_returns.iloc[len(recent_returns) - 1]
    signal = pd.DataFrame((rebal_date.rank(axis=1, ascending = False) <= selected_num).applymap(lambda x : '1' if x == True else '0'))
    signal = signal.shift(1).fillna(0)
    signal = signal.astype(float)
    return signal

rm_signal_m = get_rm_signal_m(df, lookback_m,selected_num)

# 3. 수익률 산출하기

def get_rm_return(df,signal, selected_num) :
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
    rm_signal = book[TICKER].astype(float)
    df = df[['Date', 'XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']]
    df.set_index(['Date'],inplace=True)
    df = df.pct_change().fillna(0)
    result = pd.DataFrame(((book * df) * 1 / selected_num).sum(axis=1))
    return result, rm_signal

rm_signal = get_rm_return(df,rm_signal_m, selected_num)[1]

def get_am_signal(df,lookback_d) :
    df_hrtn = df.pct_change(lookback_d).fillna(0)
    am_signal = (df_hrtn > 0).applymap(lambda x : '1' if x == 1 else '0')
    am_signal = am_signal.astype(float)
    return am_signal

am_signal = get_am_signal(df,lookback_d)

def get_dm_signal(rm_signal, am_signal):
    dm_signal = am_signal * rm_signal
    dm_signal = dm_signal.astype(float)
    return dm_signal

dm_signal = get_dm_signal(rm_signal, am_signal)

def get_dm_return(df, dm_signal):
    df_rtn = df.pct_change().fillna(0)
    result = (df_rtn * dm_signal).sum(axis=1) * (1 / dm_signal.sum(axis=1)).replace([np.inf, -np.inf], np.nan).fillna(1)
    return result

dm_result = get_dm_return(df, dm_signal)


def get_mdd( x ):
    arr = np.array( x )
    idx_lower = np.argmin( arr - np.maximum.accumulate( arr ) )
    idx_upper = np.argmax( arr[ :idx_lower ] )
    MDD = ( arr[ idx_lower ] - arr[ idx_upper ] ) / arr[ idx_upper ]
    return MDD
def get_cagr( x ):
    CAGR = ((x[len( x )-1] + 1) ** (252 / len( x )) - 1)
    return CAGR
def get_vol ( x ):
    VOL =  np.std( x ) * np.sqrt(252)
    return VOL
def get_srp ( x ):
    SRP =  np.mean( x ) / np.std( x )  * np.sqrt(252)
    return SRP


df_mdd = get_mdd(bm['Adj Close'].iloc[:]) * 100
# df_cagr = get_cagr(dm_result.iloc[:]) * 100
df_cagr = get_cagr(((1+dm_result).cumprod()-1).iloc[:]) * 100
df_vol = get_vol(dm_result.iloc[:]) * 100
df_srp = get_srp(dm_result.iloc[:])


result = pd.DataFrame(data = np.array([df_mdd,df_cagr,df_vol,df_srp]), index = ['MDD','CAGR','VOL','SRP'], columns = ['RM_Momentum'])

dm_signal * (1 / dm_signal.sum(axis=1)).replace([np.inf, -np.inf], np.nan).fillna(1)



def make_colors(n, colormap=plt.cm.Spectral):
    return colormap(np.linspace(0.1, 1.0, n))

# Cumulative Compounded Returns for RelativeMomentum
plt.figure(figsize=(17,7))
plt.title('Dual Momentum Return')
plt.plot((1 + dm_result).cumprod() - 1, label = 'DM_Momentum')
plt.plot((1 + bm['Adj Close'].pct_change().fillna(0)).cumprod() - 1, label = 'BenchMark')
plt.legend()
plt.show()

# Cross-Sectional Weights
plt.figure(figsize=(17,7))
plt.title('Cross-Sectional Weights')
plt.stackplot(dm_signal.index, np.transpose(dm_signal),labels = dm_signal.columns, colors=make_colors(len(dm_signal.columns)))
plt.legend()
plt.show()