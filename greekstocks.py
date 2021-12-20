# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:42:37 2021

@author: IOANNIS PSOMIADIS
"""
import warnings
from datetime import datetime
import csv
from contextlib import closing
import pandas as pd
import numpy as np
from scipy import stats
import requests as req
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt.risk_models import CovarianceShrinkage
from stock_tickers import tickers_gr


def get_greekstocks_data(tickers):
    '''get historical stock prices from naftemporiki site'''
    stocks_data = {}
    for ticker in tickers:
        dates = []
        open_ = []
        high = []
        low = []
        close = []
        volume = []
        url = f'https://www.naftemporiki.gr/finance/Data/getHistoryData.aspx?symbol={ticker}&type=csv'
        with closing(req.get(url, verify=True, stream=True)) as ret:
            f = (line.decode('utf-8') for line in ret.iter_lines())
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                dates.append(row[0])
                row[1] = row[1].replace(',', '.')
                open_.append(row[1])
                row[2] = row[2].replace(',', '.')
                high.append(row[2])
                row[3] = row[3].replace(',', '.')
                low.append(row[3])
                row[4] = row[4].replace(',', '.')
                close.append(row[4])
                row[5] = row[5].replace(',', '.')
                volume.append(row[5])
        del dates[0]
        del open_[0]
        del high[0]
        del low[0]
        del close[0]
        del volume[0]
        df_temp = pd.DataFrame({'date': dates, 'open': open_, 'high': high, 'low': low,
                                'close': close, 'volume': volume})
        df_temp.iloc[:, 1] = df_temp.iloc[:, 1].astype(float)
        df_temp['date'] = pd.to_datetime(df_temp['date'], format="%d/%m/%Y")
        df_temp.iloc[:, 1:] = df_temp.iloc[:, 1:].astype(float)
        data = df_temp.reset_index(drop=True)
        stocks_data[ticker] = data
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
        ret = np.log(1 + prices.pct_change())
    else:
        ret = prices.pct_change()
    return ret


def capm_returns(prices, market_prices=None, returns_data=False, risk_free_rate=0.02,
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


def clean_weights(weights, cutoff=1e-3, rounding=3):
    '''clean the weights of near zero values'''
    if weights is None:
        raise AttributeError("Weights not yet computed")
    cleaned_weights = weights.copy()
    cleaned_weights[np.abs(cleaned_weights) < cutoff] = 0
    if rounding is not None:
        if not isinstance(rounding, int) or rounding < 1:
            raise ValueError("rounding must be a positive integer")
        cleaned_weights = np.round(cleaned_weights, rounding)
    return cleaned_weights


def momentum_score(ts_):
    ''' Momentum score function. This is a long only indicator based on the annualized \
        slope of the price'''
    x = np.arange(len(ts_))
    log_ts = np.log(ts_)
    regress = stats.linregress(x, log_ts)
    annualized_slope = (np.power(np.exp(regress[0]), 252) - 1) * 100
    return annualized_slope * (regress[2] ** 2)


def select_columns(data_frame, column_names):
    '''select columns from a dataframe'''
    new_frame = data_frame.loc[:, column_names]
    return new_frame


def get_portfolio(universe, df_tr, port_value, cutoff, df_m):
    '''create a portfolio using the stocks from the universe and the closing
    prices from df_tr with a given portfolio value and a weight cutoff value
    using the value of a momemntum indicator to limit the quantity of the stocks'''
    df_t = select_columns(df_tr, universe)
    mu = capm_returns(df_t)
    S = CovarianceShrinkage(df_t).ledoit_wolf()
    # Optimize the portfolio for min volatility
    ef = EfficientFrontier(mu, S)  # Use regularization (gamma=1)
    weights = ef.min_volatility()
    #weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights(cutoff=cutoff)
    # Allocate
    latest_prices = get_latest_prices(df_t)
    da = DiscreteAllocation(cleaned_weights,
                            latest_prices,
                            total_portfolio_value=port_value
                            )
    allocation = da.greedy_portfolio()[0]
    non_trading_cash = da.greedy_portfolio()[1]
    # Put the stocks and the number of shares from the portfolio into a df
    symbol_list = []
    mom = []
    w = []
    num_shares_list = []
    l_price = []
    tot_cash = []
    for symbol, num_shares in allocation.items():
        symbol_list.append(symbol)
        mom.append(df_m[df_m['stock'] == symbol].values[0])
        w.append(round(cleaned_weights[symbol], 4))
        num_shares_list.append(num_shares)
        l_price.append(latest_prices[symbol])
        tot_cash.append(num_shares*latest_prices[symbol])

    df_buy = pd.DataFrame()
    df_buy['stock'] = symbol_list
    df_buy['momentum'] = mom
    df_buy['weights'] = w
    df_buy['shares'] = num_shares_list
    df_buy['price'] = l_price
    df_buy['value'] = tot_cash
    df_buy = df_buy.append({'stock': 'CASH', 'momentum': 0, 'weights': round(1-df_buy['weights'].sum(
    ), 4), 'shares': 1, 'price': round(non_trading_cash, 2), 'value': round(non_trading_cash, 2)}, ignore_index=True)
    df_buy = df_buy.set_index('stock')
    return df_buy, non_trading_cash


def backtest_portfolio(prices_df, bt_dataset=800, lookback_days=700, momentum_window=120, minimum_momentum=70, portfolio_size=5, tr_period=5, cutoff=0.05, port_value=10000, a_v=0):
    print(bt_dataset, lookback_days, momentum_window, minimum_momentum,
          portfolio_size, tr_period, cutoff, port_value, a_v)
    allocation = {}
    non_trading_cash = 0
    added_value = tr_period*a_v
    no_tr = 0  # number of trades performed
    init_portvalue = port_value
    eff = 1
    plotted_portval = []
    plotted_ret = []
    pval = pd.DataFrame(columns=['Date', 'portvalue', 'porteff'])
    # boolean to see if we have an optimized portfolio during a trading period
    keep_df_buy = True
    for days in range(bt_dataset, len(prices_df)+1, tr_period):
        # sliced dataset for the trading period
        df_tr = prices_df.iloc[days-lookback_days:days, :]
        df_date = datetime.strftime(prices_df.iloc[days, :].name, '%d-%m-%Y')
        if days <= bt_dataset:
            ini_date = df_date
            plotted_portval.append(round(port_value, 2))
            plotted_ret.append(round(100*eff, 2))
            pval = pval.append({'Date': df_date, 'portvalue': round(
                init_portvalue, 2), 'porteff': round(0, 2)}, ignore_index=True)
        elif days > bt_dataset and keep_df_buy is False:
            latest_prices = get_latest_prices(df_tr)
            allocation = df_buy['shares'].to_dict()
            new_port_value = non_trading_cash
            for s in list(allocation.keys())[:-1]:
                new_port_value = new_port_value + \
                    allocation.get(s)*latest_prices.get(s)
            eff = new_port_value/port_value-1
            port_value = new_port_value
            port_value = port_value+added_value  # add 200 after each trading period
        df_m = pd.DataFrame()
        m_s = []
        st = []
        for s in tickers_gr:
            st.append(s)
            m_s.append(momentum_score(df_tr[s].tail(momentum_window)))
        df_m['stock'] = st
        df_m['momentum'] = m_s
        dev = df_m['momentum'].std()
        # Get the top momentum stocks for the period
        df_m = df_m.sort_values(by='momentum', ascending=False)
        if len(df_m[(df_m['momentum'] > minimum_momentum-0.5*dev) & (df_m['momentum'] < minimum_momentum+1.9*dev)]) < portfolio_size:
            df_m = df_m.head(portfolio_size)
        else:
            df_m = df_m[(df_m['momentum'] > minimum_momentum-0.5*dev) &
                        (df_m['momentum'] < minimum_momentum+1.9*dev)].head(portfolio_size)
        # Set the universe to the top momentum stocks for the period
        universe = df_m['stock'].tolist()
        # Create a df with just the stocks from the universe
        if port_value > 0:
            keep_df_buy = False
            df_buy, non_trading_cash = get_portfolio(
                universe, df_tr, port_value, cutoff, df_m)
            port_value = df_buy['value'].sum()
            plotted_portval.append(round(port_value, 2))
            plotted_ret.append(round(100*eff, 2))
            pval = pval.append({'Date': df_date, 'portvalue': round(
                port_value, 2), 'porteff': round(eff*100, 2)}, ignore_index=True)
            no_tr = no_tr+1
        else:
            port_value = port_value+added_value
            plotted_portval.append(round(port_value, 2))
            plotted_ret.append(round(100*eff, 2))
            pval = pval.append({'Date': df_date, 'portvalue': round(
                port_value, 2), 'porteff': round(eff*100, 2)}, ignore_index=True)
            keep_df_buy = True
    total_ret = 100*(new_port_value/(init_portvalue+no_tr*added_value)-1)
    tot_contr = init_portvalue+no_tr*added_value
    s = round(pd.DataFrame(plotted_portval).pct_change().add(1).cumprod()*100, 2)
    rs = {'trades': no_tr,
          'momentum_window': momentum_window,
          'minimum_momentum': minimum_momentum,
          'portfolio_size': portfolio_size,
          'tr_period': tr_period,
          'cutoff': cutoff,
          'tot_contribution': tot_contr,
          'final port_value': new_port_value,
          'cumprod': s[-1:][0].values[0],
          'tot_ret': total_ret,
          'drawdown': s.diff().min()[0]}
    plt.plot(plotted_portval)
    plt.title('Portfolio Value history')
    plt.xlabel('Trades')
    plt.ylabel('Portfolio Value')
    plt.show()
    return rs


def backtest_portfolio2(prices_df, bt_dataset=800, lookback_days=700, momentum_window=120, minimum_momentum=70, portfolio_size=5, tr_period=5, cutoff=0.05, port_value=10000, a_v=0):
    print(bt_dataset, lookback_days, momentum_window, minimum_momentum,
          portfolio_size, tr_period, cutoff, port_value, a_v)
    allocation = {}
    non_trading_cash = 0
    added_value = tr_period*a_v
    no_tr = 0  # number of trades performed
    init_portvalue = port_value
    eff = 1
    plotted_portval = []
    plotted_ret = []
    pval = pd.DataFrame(columns=['Date', 'portvalue', 'porteff'])
    # boolean to see if we have an optimized portfolio during a trading period
    keep_df_buy = True
    for days in range(bt_dataset, len(prices_df), tr_period):
        # sliced dataset for the trading period
        df_tr = prices_df.iloc[days-lookback_days:days, :]
        df_date = datetime.strftime(prices_df.iloc[days, :].name, '%d-%m-%Y')
        print(plotted_portval)
        if days <= bt_dataset:
            ini_date = df_date
            plotted_portval.append(round(port_value, 2))
            plotted_ret.append(round(100*eff, 2))
            pval = pval.append({'Date': df_date, 'portvalue': round(
                init_portvalue, 2), 'porteff': round(0, 2)}, ignore_index=True)
        elif days > bt_dataset and keep_df_buy is False:
            latest_prices = get_latest_prices(df_tr)
            allocation = df_buy['shares'].to_dict()
            new_port_value = non_trading_cash
            for s in list(allocation.keys())[:-1]:
                new_port_value = new_port_value + \
                    allocation.get(s)*latest_prices.get(s)
                print('Sell ', s, 'stocks: ', allocation.get(s), ' bought for ', df_buy['price'][s], ' sold for ', latest_prices.get(s), ' for total:{0:.2f} and a gain of :{1:.2f}'.format(allocation.get(s)*latest_prices.get(s),
                      (latest_prices.get(s)-df_buy['price'][s])*allocation.get(s)))
            eff = new_port_value/port_value-1
            print('Return after trading period {0:.2f}%  for a total Value {1:.2f}'.format(
                eff*100, new_port_value))
            port_value = new_port_value
            port_value = port_value+added_value
        print(df_tr.iloc[[0, -1], :3])
        df_m = pd.DataFrame()
        m_s = []
        st = []
        for s in tickers_gr:
            st.append(s)
            m_s.append(momentum_score(df_tr[s].tail(momentum_window)))
        df_m['stock'] = st
        df_m['momentum'] = m_s
        dev = df_m['momentum'].std()
        # Get the top momentum stocks for the period
        df_m = df_m.sort_values(by='momentum', ascending=False)
        if len(df_m[(df_m['momentum'] > minimum_momentum-0.5*dev) & (df_m['momentum'] < minimum_momentum+1.9*dev)]) < portfolio_size:
            df_m = df_m.head(portfolio_size)
        else:
            df_m = df_m[(df_m['momentum'] > minimum_momentum-0.5*dev) &
                        (df_m['momentum'] < minimum_momentum+1.9*dev)].head(portfolio_size)

        # Set the universe to the top momentum stocks for the period
        universe = df_m['stock'].tolist()
        # Create a df with just the stocks from the universe
        if port_value > 0:
            keep_df_buy = False
            df_buy, non_trading_cash = get_portfolio(
                universe, df_tr, port_value, cutoff, df_m)
            print('Buy date', df_date)
            print(df_buy)
            print('trade no:', no_tr, ' non allocated cash:{0:.2f}'.format(
                non_trading_cash), 'total invested:', df_buy['value'].sum())
            port_value = df_buy['value'].sum()
            plotted_portval.append(round(port_value, 2))
            plotted_ret.append(round(100*eff, 2))
            pval = pval.append({'Date': df_date, 'portvalue': round(
                port_value, 2), 'porteff': round(eff*100, 2)}, ignore_index=True)
            no_tr = no_tr+1
        else:
            print('Buy date', df_date,
                  'Not enough money to create portfolio', port_value)
            port_value = port_value+added_value
            plotted_portval.append(round(port_value, 2))
            plotted_ret.append(round(100*eff, 2))
            pval = pval.append({'Date': df_date, 'portvalue': round(
                port_value, 2), 'porteff': round(eff*100, 2)}, ignore_index=True)
            keep_df_buy = True

    total_ret = 100*(new_port_value/(init_portvalue+no_tr*added_value)-1)
    duration = no_tr*tr_period
    print('Total return: {0:.2f}% in {1} days'.format(total_ret, duration))
    print('Cumulative portfolio return:', round(
        list(pval['porteff'].cumsum())[-1], 2))
    print('total capital:', init_portvalue+no_tr*added_value, new_port_value)
    tot_contr = init_portvalue+no_tr*added_value
    s = round(pd.DataFrame(plotted_portval).pct_change().add(1).cumprod()*100, 2)
    rs = {'trades': no_tr,
          'momentum_window': momentum_window,
          'minimum_momentum': minimum_momentum,
          'portfolio_size': portfolio_size,
          'tr_period': tr_period,
          'cutoff': cutoff,
          'tot_contribution': tot_contr,
          'final port_value': new_port_value,
          'cumprod': s[-1:][0].values[0],
          'tot_ret': total_ret,
          'drawdown': s.diff().min()[0]}
    plt.plot(plotted_portval)
    plt.title('Portfolio Value history')
    plt.xlabel('Trades')
    plt.ylabel('Portfolio Value')
    plt.show()
    return rs


def rebalance_portfolio(df_old, df_new):
    '''rebalance old with new proposed portfolio'''

    new_stocks = list(df_old.stock[:-1]) + \
        list(set(df_new.stock[:-1])-set(df_old.stock))
    for stock in new_stocks:
        # close old positions that do not appear in new portfolio
        if (stock in list(df_old.stock)) and (stock not in list(df_new.stock[:-1])):
            # close positions
            if df_old.loc[df_old.stock == stock, 'shares'].values[0] > 0:
                print('Close position on ', stock)

            if df_old.loc[df_old.stock == stock, 'shares'].values[0] < 0:
                print('Close position on ', stock)

        # open new positions that only appear in new portfolio
        if stock in list(set(df_new.stock[:-1])-set(df_old.loc[:, 'stock'])):
            if df_new.loc[df_new.stock == stock, 'shares'].values[0] > 0:
                print('Buy ', df_new.loc[df_new.stock == stock, 'shares'].values[0], ' shares of ',
                      stock, ' to open new long position')

            if df_new.loc[df_new.stock == stock, 'shares'].values[0] < 0:
                print('Sell ', df_new.loc[df_new.stock == stock, 'shares'].values[0], ' shares of ',
                      stock, ' to open new short position')

        # modify positions of stocks that appear in new and old portfolio
        if (stock in list(df_old.stock)) and (stock in list(df_new.stock[:-1])):
            # change positions
            if df_new.loc[df_new.stock == stock, 'shares'].values[0] > 0 and df_old.loc[df_old.stock == stock, 'shares'].values[0] > 0:
                new_shares = df_new.loc[df_new.stock == stock, 'shares'].values[0] - \
                    df_old.loc[df_old.stock == stock, 'shares'].values[0]
                if new_shares >= 0:
                    print('Increase position of', round(
                        new_shares, 4), ' shares on ', stock)

                if new_shares < 0:
                    print('Reduce position of ', round(
                        new_shares, 4), ' shares on ', stock)
            if df_new.loc[df_new.stock == stock, 'shares'].values[0] < 0 and df_old.loc[df_old.stock == stock, 'shares'].values[0] < 0:
                new_shares = df_new.loc[df_new.stock == stock, 'shares'].values[0] - \
                    df_old.loc[df_old.stock == stock, 'shares'].values[0]
                if new_shares >= 0:
                    print('Buy another ', round(new_shares, 4), ' of ', stock)
                if new_shares < 0:
                    print('Sell another ', round(new_shares, 4), ' of ', stock)
            if df_new.loc[df_new.stock == stock, 'shares'].values[0]*df_old.loc[df_old.stock == stock, 'shares'].values[0] < 0:
                new_shares = df_new.loc[df_new.stock == stock, 'shares'].values[0] - \
                    df_old.loc[df_old.stock == stock, 'shares'].values[0]
                if new_shares >= 0:
                    print('Buy Long', round(new_shares, 4), ' of ', stock)
                if new_shares < 0:
                    print('Sell Short ', round(new_shares, 4), ' of ', stock)
    return df_new['value'].sum()
