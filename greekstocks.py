# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:42:37 2021

@author: IOANNIS PSOMIADIS
"""
import warnings
import csv
from contextlib import closing
import pandas as pd
import numpy as np
from scipy import stats
import requests as req


def get_greekstocks_data(tickers):
    '''get historical stock prices from naftemporiki site'''
    stocks_data={}
    for ticker in tickers:
        dates=[]
        open_=[]
        high=[]
        low=[]
        close=[]
        volume=[]
        url=f'https://www.naftemporiki.gr/finance/Data/getHistoryData.aspx?symbol={ticker}&type=csv'
        with closing(req.get(url, verify=True, stream=True)) as ret:
            f = (line.decode('utf-8') for line in ret.iter_lines())
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                dates.append(row[0])
                row[1]=row[1].replace(',','.')
                open_.append(row[1])
                row[2]=row[2].replace(',','.')
                high.append(row[2])
                row[3]=row[3].replace(',','.')
                low.append(row[3])
                row[4]=row[4].replace(',','.')
                close.append(row[4])
                row[5]=row[5].replace(',','.')
                volume.append(row[5])
        del dates[0]
        del open_[0]
        del high[0]
        del low[0]
        del close[0]
        del volume[0]
        df_temp=pd.DataFrame({'date':dates, 'open':open_, 'high':high,'low':low, \
                              'close':close,'volume':volume})
        df_temp.iloc[:,1]=df_temp.iloc[:,1].astype(float)
        df_temp['date'] =pd.to_datetime(df_temp['date'],format="%d/%m/%Y")
        df_temp.iloc[:,1:]=df_temp.iloc[:,1:].astype(float)
        data=df_temp.reset_index(drop=True)#
        stocks_data[ticker]=data
    return stocks_data


def returns_from_prices(prices, log_returns=False):
    """
    Calculate the returns given prices.
    :param prices: adjusted (daily) closing prices of the asset, each row is a
                   date and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param log_returns: whether to compute using log returns
    :type log_returns: bool, defaults to False
    :return: (daily) returns
    :rtype: pd.DataFrame
    """
    if log_returns:
        ret= np.log(1 + prices.pct_change())
    else:
        ret= prices.pct_change()
    return ret


def capm_returns(prices, market_prices=None, returns_data=False, risk_free_rate=0.02, \
                 compounding=True, frequency=252):
    """
    Compute a return estimate using the Capital Asset Pricing Model. Under the CAPM,
    asset returns are equal to market returns plus a :math:`\beta` term encoding
    the relative risk of the asset.
    .. math::
        R_i = R_f + \\beta_i (E(R_m) - R_f)
    :param prices: adjusted closing prices of the asset, each row is a date
                    and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param market_prices: adjusted closing prices of the benchmark, defaults to None
    :type market_prices: pd.DataFrame, optional
    :param returns_data: if true, the first arguments are returns instead of prices.
    :type returns_data: bool, defaults to False.
    :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.
                           You should use the appropriate time period, corresponding
                           to the frequency parameter.
    :type risk_free_rate: float, optional
    :param compounding: computes geometric mean returns if True,
                        arithmetic otherwise, optional.
    :type compounding: bool, defaults to True
    :param frequency: number of time periods in a year, defaults to 252 (the number
                        of trading days in a year)
    :type frequency: int, optional
    :return: annualised return estimate
    :rtype: pd.Series
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
        market_returns = market_prices
    else:
        returns = returns_from_prices(prices)
        if market_prices is not None:
            market_returns = returns_from_prices(market_prices)
        else:
            market_returns = None
    # Use the equally-weighted dataset as a proxy for the market
    if market_returns is None:
        # Append market return to right and compute sample covariance matrix
        returns["mkt"] = returns.mean(axis=1)
    else:
        market_returns.columns = ["mkt"]
        returns = returns.join(market_returns, how="left")
    # Compute covariance matrix for the new dataframe (including markets)
    cov = returns.cov()
    # The far-right column of the cov matrix is covariances to market
    betas = cov["mkt"] / cov.loc["mkt", "mkt"]
    betas = betas.drop("mkt")
    # Find mean market return on a given time period
    if compounding:
        mkt_mean_ret = (1 + returns["mkt"]).prod() ** (
            frequency / returns["mkt"].count()
        ) - 1
    else:
        mkt_mean_ret = returns["mkt"].mean() * frequency
    # CAPM formula
    return risk_free_rate + betas * (mkt_mean_ret - risk_free_rate)


def get_latest_prices(prices):
    '''get latest prices of all stocks'''
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices not in a dataframe")
    return prices.ffill().iloc[-1]


def momentum_score(ts):
    ''' Momentum score function. This is a long only indicator based on the annualized \
        slope of the price'''
    x = np.arange(len(ts))
    log_ts = np.log(ts)
    regress = stats.linregress(x, log_ts)
    annualized_slope = (np.power(np.exp(regress[0]), 252) -1) * 100
    return annualized_slope * (regress[2] ** 2)


def select_columns(data_frame, column_names):
    '''select columns from a dataframe'''
    new_frame = data_frame.loc[:, column_names]
    return new_frame


def rebalance_portfolio(df_old,df_new):
    '''rebalance old with new proposed portfolio'''
    old_port_value=df_old['value'].sum()
    new_port_value=old_port_value
    new_stocks= list(df_old.stock[:-1]) + list(set(df_new.stock[:-1])-set(df_old.stock))
    for stock in new_stocks:
        #close old positions that do not appear in new portfolio
        if (stock in list(df_old.stock)) and (stock not in list(df_new.stock[:-1])) :
            #close positions
            if df_old.loc[df_old.stock==stock,'shares'].values[0]>0:
                print('κλείσιμο θέσης στην μετοχ΄ή ',stock)
                new_port_value=new_port_value+df_old.loc[df_old.stock==stock,'shares'].values[0]
            if df_old.loc[df_old.stock==stock,'shares'].values[0]<0:
                print('κλείσιμο θέσης στην μετοχ΄ή ',stock)
                new_port_value=new_port_value+df_old.loc[df_old.stock==stock,'shares'].values[0]
        #open new positions that only appear in new portfolio
        if stock in list(set(df_new.stock[:-1])-set(df_old.loc[:,'stock'])):
            if df_new.loc[df_new.stock==stock,'shares'].values[0]>0:
                print('Buy ',df_new.loc[df_new.stock==stock,'shares'].values[0],' shares of ',
                      stock,' to open long position')
            if df_new.loc[df_new.stock==stock,'shares'].values[0]<0:
                print('Sell ',df_new.loc[df_new.stock==stock,'shares'].values[0],' shares of ',
                      stock,' to open short position')
        #modify positions of stocks that appear in new and old portfolio
        if (stock in list(df_old.stock)) and (stock in list(df_new.stock[:-1])):
            #change positions
            if df_new.loc[df_new.stock==stock,'shares'].values[0]>0 and df_old.loc[df_old.stock==stock,'shares'].values[0]>0:
                new_shares=df_new.loc[df_new.stock==stock,'shares'].values[0]-df_old.loc[df_old.stock==stock,'shares'].values[0]
                if new_shares>=0:
                    print('Buy another ',round(new_shares,4),' of ',stock)
                if new_shares<0:
                    print('Sell another ',round(new_shares,4),' of ',stock)
            if df_new.loc[df_new.stock==stock,'shares'].values[0]<0 and df_old.loc[df_old.stock==stock,'shares'].values[0]<0:
                new_shares=df_new.loc[df_new.stock==stock,'shares'].values[0]-df_old.loc[df_old.stock==stock,'shares'].values[0]
                if new_shares>=0:
                    print('Buy another ',round(new_shares,4),' of ',stock)
                if new_shares<0:
                    print('Sell another ',round(new_shares,4),' of ',stock)
            if df_new.loc[df_new.stock==stock,'shares'].values[0]*df_old.loc[df_old.stock==stock,'shares'].values[0] < 0:
                new_shares=df_new.loc[df_new.stock==stock,'shares'].values[0] - df_old.loc[df_old.stock==stock,'shares'].values[0]
                if new_shares>=0:
                    print('Buy Long',round(new_shares,4),' of ',stock)
                if new_shares<0:
                    print('Sell Short ',round(new_shares,4),' of ',stock)
    return new_port_value             