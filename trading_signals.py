# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:42:37 2021

@author: PSOMIADIS
"""

import numpy as np
import talib


def get_signals(data,ticker):
    data['atr']=(talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)).round(4)
    data['rsi']=talib.RSI(data['close'], timeperiod=14)
    data['adx']=(talib.ADX(data['high'], data['low'], data['close'], timeperiod=21)).round(4)
    data['DIpos']=(talib.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=21)).round(4)
    data['DIneg']=(talib.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=21)).round(4)
    data['ema200']=(talib.EMA(data['close'], timeperiod=200))
    data['macd'], data['macdsignal'], data['macdhist'] =(talib.MACD(data['close'], fastperiod=5, slowperiod=21, signalperiod=13))
    data['pSAR']=(talib.SAR(data['high'],data['low'], acceleration=0.01, maximum=0.2)).round(6)
    data['lreg']=(talib.LINEARREG(data['close'], timeperiod=14)).round(6)
    data['volclose']=data['close']*data['volume']
    data['VWMA']=(data['volclose'].rolling(14).sum()/ data['volume'].rolling(14).sum()).round(6)
    data['rsivol']=talib.RSI(data['rsi']/data['atr'], timeperiod=14)
    data['sma13']=talib.SMA(data['close'], timeperiod=13)
    data['rsisma']=talib.RSI(data['sma13'], timeperiod=13)
    data['sma14']=talib.SMA(data['close'], timeperiod=14)
    data['DI14']=((data['close']-data['sma14'])/ data['sma14'])*100
    data['rocsma1']=(talib.SMA(talib.ROC(data['close'], timeperiod=10), timeperiod=10))
    data['rocsma2']=(talib.SMA(talib.ROC(data['close'], timeperiod=15), timeperiod=10))
    data['rocsma3']=(talib.SMA(talib.ROC(data['close'], timeperiod=20), timeperiod=10))
    data['rocsma4']=(talib.SMA(talib.ROC(data['close'], timeperiod=30), timeperiod=15))
    data['KST']=data['rocsma1']+ 2*data['rocsma2']+ 3*data['rocsma3']+ 4*data['rocsma4']
    data['KSTsignal']=talib.SMA(data['KST'],timeperiod=9)
    data['PC']=data['close']- data['close'].shift(1)
    data['AbsPC']=abs(data['PC'])
    data['dsmPC'] =talib.EMA(talib.EMA(data['PC'], timeperiod=25), timeperiod=13)
    data['dsmAPC']=talib.EMA(talib.EMA(data['AbsPC'], timeperiod=25), timeperiod=13)
    data['TSILine']=100*data['dsmPC']/ data['dsmAPC']
    data['SignalTSI']=talib.EMA(data['TSILine'], timeperiod=13)
    data['sdouble']=np.where((data['close']> data['pSAR'])&(data['rsi']> 50),1,np.where((data['close']< data['pSAR'])&(data['rsi']< 50), -1, 0))
    data['sRSIVOL']=np.where((data['rsivol']< 20), 1, np.where((data['rsivol']> 80), -1, 0))
    data['sDI14']=np.where((data['DI14']>0), 1, np.where((data['DI14']<0), -1, 0))
    data['sKST']=np.where((data['KST']>data['KSTsignal']), 1, np.where((data['KST']< data['KSTsignal']), -1, 0))
    data['sTSILine']=np.where((data['TSILine']> data['SignalTSI']), 1, np.where((data['TSILine']< data['SignalTSI']), -1, 0))
    data['sDIKSTTSI']=np.where((data['DI14']> 0)&(data['KST']> data['KSTsignal'])&(data['TSILine']> data['SignalTSI']), 1,
                             np.where((data['DI14']< 0)&(data['KST']< data['KSTsignal'])&(data['TSILine']< data['SignalTSI']), -1, 0))
    data['sRSIMAma']=np.where((data['rsisma'].shift(1)< 10)&(data['rsisma']> 10), 1,
                             np.where((data['rsisma'].shift(1)> 90)&(data['rsisma']< 90), -1, 0))
    data['sMACD']=np.where(((data['macd']> data['macdsignal'])&(data['macd']< 0)&(data['macdsignal']< 0)), 1, np.where(((data['macd']< data['macdsignal'])&(data['macd']> 0)&(data['macdsignal']> 0)), -1, 0))
    data['sEMA200']=np.where((data['close']> data['ema200']), 1, np.where((data['close']< data['ema200']), -1, 0))
    data['sSAR']=np.where((data['close']>data['pSAR']), 1, np.where((data['close']< data['pSAR']), -1, 0))
    data['sADX']=np.where((data['adx']>25)&(data['DIpos']>data['DIneg']),1,np.where((data['adx']<25)&(data['DIpos']<data['DIneg']),-1,0))
    data['sMAMACDSAR']=np.where((data['sMACD']==1)&(data['sEMA200']==1)&(data['sSAR']==1),1,np.where((data['sMACD']==-1)&(data['sEMA200']==-1)&(data['sSAR']==-1),-1,0))
    data['sLRVWMA']=np.where((data['lreg']>data['VWMA']),1,np.where((data['lreg']<data['VWMA']),-1,0))
    return data.iloc[-1,-13:]
