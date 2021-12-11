# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:42:37 2021

@author: IOANNIS PSOMIADIS
"""
#import streamlit
import warnings
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt.risk_models import CovarianceShrinkage
from greekstocks import get_greekstocks_data, capm_returns, get_latest_prices, momentum_score
from greekstocks import select_columns, rebalance_portfolio


# list of selected greek stocks tickers , 'NEWS.ATH', 'PPAK.ATH','TITC.ATH','CNLCAP.ATH',
tickers_gr = ['AEGN.ATH', 'AETF.ATH', 'ALMY.ATH', 'ALPHA.ATH', 'ANDRO.ATH', 'ANEK.ATH',
              'ASCO.ATH', 'ASTAK.ATH', 'ATEK.ATH', 'ATRUST.ENAX', 'ATTICA.ATH', 'AVAX.ATH',
              'AVE.ATH', 'BELA.ATH', 'BIOKA.ATH', 'BIOSK.ATH', 'BIOT.ATH', 'BRIQ.ATH',
              'BYTE.ATH', 'CENER.ATH',  'CRETA.ATH', 'DAIOS.ATH', 'DOMIK.ATH', 'DROME.ATH',
              'DUR.ATH', 'EEE.ATH', 'EKTER.ATH', 'ELBE.ATH', 'ELBIO.ATH', 'ELHA.ATH',
              'ELIN.ATH', 'ELLAKTOR.ATH', 'ELPE.ATH', 'ELSTR.ATH', 'ELTON.ATH', 'ENTER.ATH',
              'EPSIL.ATH', 'ETE.ATH', 'EUPIC.ATH', 'EUROB.ATH', 'EVROF.ATH', 'EXAE.ATH',
              'EYAPS.ATH', 'EYDAP.ATH', 'FIER.ATH', 'FLEXO.ATH', 'FOODL.ENAX', 'FOYRK.ATH',
              'GEBKA.ATH', 'GEKTERNA.ATH', 'HTO.ATH', 'IATR.ATH', 'IKTIN.ATH', 'ILYDA.ATH',
              'INKAT.ATH', 'INLOT.ATH', 'INTERCO.ATH', 'INTET.ATH', 'INTRK.ATH', 'KAMP.ATH',
              'KEKR.ATH', 'KEPEN.ATH', 'KLM.ATH', 'KMOL.ATH', 'KORDE.ATH', 'KREKA.ATH',
              'KRI.ATH', 'KTILA.ATH', 'KYLO.ATH', 'KYRI.ATH', 'LAMDA.ATH', 'LANAC.ATH',
              'LAVI.ATH', 'LEBEK.ATH', 'LOGISMOS.ATH', 'LYK.ATH', 'MATHIO.ATH', 'MEDIC.ATH',
              'MERKO.ATH', 'MEVA.ATH', 'MIG.ATH', 'MIN.ATH', 'MOH.ATH', 'MYTIL.ATH',
              'OLTH.ATH', 'OLYMP.ATH', 'OPAP.ATH', 'OTOEL.ATH', 'PAIR.ATH', 'PAP.ATH',
              'PERF.ENAX', 'PETRO.ATH', 'PLAIS.ATH', 'PLAKR.ATH', 'PLAT.ATH', 'PPA.ATH',
              'PROF.ATH', 'QUAL.ATH', 'QUEST.ATH', 'REVOIL.ATH', 'SAR.ATH', 'SPACE.ATH',
              'SPIR.ATH', 'TATT.ATH', 'TELL.ATH', 'TENERGY.ATH',  'TPEIR.ATH', 'TRASTOR.ATH',
              'VARG.ATH', 'VARNH.ATH', 'VIDAVO.ENAX', 'VIO.ATH', 'VIS.ATH', 'VOSYS.ATH',
              'YALCO.ATH', 'ADMIE.ATH', 'PPC.ATH']

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")

# get all greek stock tickers price data
greekstocks_data = get_greekstocks_data(tickers_gr)
# create the close prices dataframe and other data
l_close = pd.DataFrame(columns=['stock', 'date', 'lastprice', 'len_prices'])
close_data = pd.DataFrame()
i = 1
for ticker in tickers_gr:
    last_close = greekstocks_data[ticker].iloc[-1]['close']
    last_date = greekstocks_data[ticker].iloc[-1]['date']
    len_values = len(greekstocks_data[ticker])
    l_close = l_close.append({'stock': ticker, 'date': last_date, 'lastprice': last_close,
                              'len_prices': len_values}, ignore_index=True)
    df_temp = greekstocks_data[ticker].loc[:, ['date', 'close']].rename(columns={'close': ticker}) \
        .set_index('date')
    if i == 1:
        close_data = df_temp
        i = i+1
    else:
        close_data = close_data.merge(df_temp, how='inner', on='date')
l_close_min = l_close['len_prices'].min()

# Declare Constants
port_value = 10000
new_port_value = 0
df = close_data
#momentum_window = v
#minimum_momentum = 70
# portfolio_size=15
# cutoff=0.1
# how much cash to add each trading period
# tr_period=21 #trading period, 21 is a month,10 in a fortnite, 5 is a week, 1 is everyday
# dataset=800 #start for length of days used for the optimising dataset
# l_days=600  #how many days to use in optimisations
res = pd.DataFrame(columns=['trades', 'momentum_window', 'minimum_momentum', 'portfolio_size',
                            'tr_period', 'cutoff', 'tot_contribution', 'final port_value',
                            'cumprod', 'tot_ret', 'drawdown'])
# total backtests 4*5*4*3*2=480
for momentum_window in [120, 252, 378, 504]:
    for minimum_momentum in [70, 100, 120, 150, 180]:
        for portfolio_size in [5, 10, 15, 20]:
            for tr_period in [5, 10, 20]:
                for cutoff in [0.05, 0.1]:
                    allocation = {}
                    dataset = 700  # start for length of days used for the optimising dataset
                    l_days = 600  # how many days to use in optimisations
                    port_value = 10000
                    non_trading_cash = 0
                    new_port_value = 0
                    print(momentum_window, minimum_momentum,
                          portfolio_size, tr_period, cutoff)
                    added_value = tr_period*0
                    no_tr = 1  # number of trades performed
                    init_portvalue = port_value
                    plotted_portval = []
                    plotted_ret = []
                    pval = pd.DataFrame(
                        columns=['Date', 'portvalue', 'porteff'])
                    keep_df_buy = True
                    for days in range(dataset, len(df), tr_period):
                        df_tr = df.iloc[days-l_days:days, :]
                        df_date = datetime.strftime(
                            df.iloc[days, :].name, '%d-%m-%Y')
                        if days <= dataset:
                            ini_date = df_date
                        if days > dataset and keep_df_buy is False:
                            latest_prices = get_latest_prices(
                                df_tr)
                            new_port_value = non_trading_cash
                            allocation = df_buy['shares'][:-1].to_dict()
                            # print(allocation)
                            if keep_df_buy is False:
                                #print('Sell date',df_date)
                                for s in tickers_gr:
                                    if s in allocation:
                                        new_port_value = new_port_value + \
                                            allocation.get(s)*latest_prices.get(s)
                                        # print('Sell ',s,'stocks: ',allocation.get(s),' bought for ',df_buy['price'][s],' sold for ',latest_prices.get(s)
                                        #       ,' for total:{0:.2f} and a gain of :{1:.2f}'.format(allocation.get(s)*latest_prices.get(s),
                                        #      (latest_prices.get(s)-df_buy['price'][s])*allocation.get(s)))
                                eff = new_port_value/port_value-1
                                #print('Return after trading period {0:.2f}%  for a total Value {1:.2f}'.format(eff*100,new_port_value))
                                port_value = new_port_value
                                plotted_portval.append(round(port_value, 2))
                                plotted_ret.append(round(eff*100, 2))
                                pval = pval.append({'Date': df_date,
                                                    'portvalue': round(port_value, 2),
                                                    'porteff': round(eff*100, 2)},
                                                    ignore_index=True)
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
                        df_m = df_m[(df_m['momentum'] > minimum_momentum-0.5*dev) & (
                            df_m['momentum'] < minimum_momentum+1.9*dev)].head(portfolio_size)
                        # Set the universe to the top momentum stocks for the period
                        universe = df_m['stock'].tolist()
                        # print('universe',universe)
                        # Create a df with just the stocks from the universe
                        if len(universe) > 2:
                            keep_df_buy = False
                            df_t = select_columns(df_tr, universe)
                            mu = capm_returns(df_t)
                            S = CovarianceShrinkage(df_t).ledoit_wolf()
                            # Optimise the portfolio for maximal Sharpe ratio
                            # Use regularization (gamma=1)
                            ef = EfficientFrontier(mu, S)
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
                                mom.append(
                                    df_m[df_m['stock'] == symbol].values[0])
                                w.append(cleaned_weights[symbol])
                                num_shares_list.append(num_shares)
                                l_price.append(latest_prices[symbol])
                                tot_cash.append(
                                    num_shares*latest_prices[symbol])

                            df_buy = pd.DataFrame()
                            df_buy['stock'] = symbol_list
                            df_buy['momentum'] = mom
                            df_buy['weights'] = w
                            df_buy['shares'] = num_shares_list
                            df_buy['price'] = l_price
                            df_buy['value'] = tot_cash
                            df_buy = df_buy.append({'stock': 'CASH',
                                                    'momentum': 0,
                                                    'weights': round(1-df_buy['value'].sum()/port_value, 2),
                                                    'shares': 0,
                                                    'price': 0,
                                                    'value': round(port_value-df_buy['value'].sum(), 2)},
                                                    ignore_index=True)
                            df_buy = df_buy.set_index('stock')
                            #print('Buy date',df_date)
                            # print(df_buy)
                            #print('trade no:',no_tr,' non allocated cash:{0:.2f}'.format(non_trading_cash),'total invested:', df_buy['value'].sum())
                            no_tr = no_tr+1
                            # st_day=st_day+tr_period
                        else:
                            #print('Buy date',df_date,'Not enough stocks in universe to create portfolio',port_value)
                            port_value = port_value+added_value
                            keep_df_buy = True

                    total_ret = 100 * \
                        (new_port_value/(init_portvalue+no_tr*added_value)-1)
                    dura_tion = (no_tr-1)*tr_period
                    if no_tr > 2:
                        #print('Total return: {0:.2f} in {1} days'.format(total_ret,dura_tion))
                        #print('Cumulative portfolio return:',round(list(pval['porteff'].cumsum())[-1],2))
                        #print('total capital:',init_portvalue+no_tr*added_value, new_port_value)
                        tot_contr = init_portvalue+no_tr*added_value
                        s = round(pd.DataFrame(
                            plotted_portval).pct_change().add(1).cumprod()*10, 2)
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
                        res = res.append(rs, ignore_index=True)
                        print(rs)
                        plt.plot(plotted_portval)
                        print(res.sort_values(by=['tot_ret']).tail(2))
print(res.sort_values(by=['tot_ret']).tail(20))
print(res.sort_values(by=['drawdown']).head(20))
best_res = res.sort_values(by=['tot_ret']).tail(1).reset_index(drop=True)
print(best_res)

# show the backtest of the best portfolio
port_value = 10000
momentum_window = int(best_res.loc[0, 'momentum_window'])
minimum_momentum = int(best_res.loc[0, 'minimum_momentum'])
portfolio_size = int(best_res.loc[0, 'portfolio_size'])
cutoff = best_res.loc[0, 'cutoff']
tr_period = int(best_res.loc[0, 'tr_period'])
# how much cash to add each trading period
added_value = tr_period*0  # how much cash to add each trading period
no_tr = 1  # number of trades performed
allocation = {}
non_trading_cash = 0
init_portvalue = port_value
plotted_portval = []
plotted_ret = []
pval = pd.DataFrame(columns=['Date', 'portvalue', 'porteff'])
keep_df_buy = True
for days in range(dataset, len(df), tr_period):
    df_tr = df.iloc[days-l_days:days, :]
    df_date = datetime.strftime(df.iloc[days, :].name, '%d-%m-%Y')
    if days <= dataset:
        ini_date = df_date
    if days > dataset and not keep_df_buy:
        latest_prices = get_latest_prices(df_tr)
        new_port_value = non_trading_cash
        allocation = df_buy['shares'][:-1].to_dict()
        # print(allocation)
        if not keep_df_buy:
            #print('Sell date',df_date)
            for s in tickers_gr:
                if s in allocation:
                    new_port_value = new_port_value + \
                        allocation.get(s)*latest_prices.get(s)
                    print('Sell ', s, 'stocks: ', allocation.get(s), ' bought for ', df_buy['price'][s], ' sold for ', latest_prices.get(s), ' for total:{0:.2f} and a gain of :{1:.2f}'.format(allocation.get(s)*latest_prices.get(s),
                          (latest_prices.get(s)-df_buy['price'][s])*allocation.get(s)))
            eff = new_port_value/port_value-1
            print('Return after trading period {0:.2f}%  for a total Value {1:.2f}'.format(
                eff*100, new_port_value))
            port_value = new_port_value
            plotted_portval.append(round(port_value, 2))
            plotted_ret.append(round(eff*100, 2))
            pval = pval.append({'Date': df_date,
                                'portvalue': round(port_value, 2),
                                'porteff': round(eff*100, 2)},
                                ignore_index=True)
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
    df_m = df_m[(df_m['momentum'] > minimum_momentum-0.5*dev) &
                (df_m['momentum'] < minimum_momentum+1.9*dev)].head(portfolio_size)
    # Set the universe to the top momentum stocks for the period
    universe = df_m['stock'].tolist()
    # print('universe',universe)
    # Create a df with just the stocks from the universe
    if len(universe) > 2:
        keep_df_buy = False
        df_t = select_columns(df_tr, universe)
        mu = capm_returns(df_t)
        S = CovarianceShrinkage(df_t).ledoit_wolf()
        # Optimise the portfolio for maximal Sharpe ratio
        ef = EfficientFrontier(mu, S)  # Use regularization (gamma=1)
        weights = ef.min_volatility()
        #weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights(cutoff)
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
            w.append(cleaned_weights[symbol])
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
        df_buy = df_buy.append({'stock': 'CASH',
                                'momentum': 0,
                                'weights': round(1-df_buy['value'].sum()/port_value, 2),
                                'shares': 0,
                                'price': 0,
                                'value': round(port_value-df_buy['value'].sum(), 2)},
                               ignore_index=True)
        df_buy = df_buy.set_index('stock')
        print('Buy date', df_date)
        print(df_buy)
        print('trade no:', no_tr, ' non allocated cash:{0:.2f}'.format(
            non_trading_cash), 'total invested:', df_buy['value'].sum())
        no_tr = no_tr+1
    else:
        print('Buy date', df_date,
              'Not enough stocks in universe to create portfolio', port_value)
        port_value = port_value+added_value
        keep_df_buy = True

total_ret = 100*(new_port_value/(init_portvalue+no_tr*added_value)-1)
dura = (no_tr-1)*tr_period
if no_tr > 2:
    print('Total return: {0:.2f} in {1} days'.format(total_ret, dura))
    print('Cumulative portfolio return:', round(
        list(pval['porteff'].cumsum())[-1], 2))
    print('total capital:', init_portvalue+no_tr*added_value, new_port_value)
    tot_contr = init_portvalue+no_tr*added_value
    s = round(pd.DataFrame(plotted_portval).pct_change().add(1).cumprod()*10, 2)
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
    res = res.append(rs, ignore_index=True)
    print(rs)
    plt.plot(plotted_portval)

# create final and new portfolio
df_old = pd.read_csv('greekstocks_portfolio.csv').iloc[:, 1:]
cash = df_old.loc[df_old['stock'] == 'CASH']['value'].values[0]
new_price = []
for symbol in df_old['stock'][:-1].values.tolist():
    new_price.append(latest_prices[symbol])
new_price.append(0)
df_old['new_prices'] = new_price
df_old['new_value'] = df_old['new_prices']*df_old['shares']
port_value = 0.8*(cash+df_old['new_value'].sum())
# port_value=2200
df_tr = close_data.tail(l_days)
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
df_m = df_m[(df_m['momentum'] > minimum_momentum-0.5*dev) &
            (df_m['momentum'] < minimum_momentum+1.9*dev)].head(portfolio_size)
# Set the universe to the top momentum stocks for the period
universe = df_m['stock'].tolist()
print(universe)
# Create a df with just the stocks from the universe
df_t = select_columns(df_tr, universe)
# portfolio
mu = capm_returns(df_t)
#mu = expected_returns.mean_historical_return(df_t)
# S=risk_models.sample_cov(df_t)
S = CovarianceShrinkage(df_t).ledoit_wolf()
# Optimise the portfolio for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)  # , gamma=1 Use regularization (gamma=1)
#weights = ef.efficient_return(sug_ret/100)
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
ppf = list(ef.portfolio_performance())
print('Portfolio from the Assets that gave signals')
print('The proposed portfolio has the below characteristics')
print('Initial Portfolio Value : '+str(port_value)+'$')
print('Sharpe Ratio: '+str(round(ppf[2], 2)))
print('Portfolio Return: '+str(round(ppf[0]*100, 2))+'%')
print('Portfolio Volatility: '+str(round(ppf[1]*100, 2))+'%')


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
    w.append(cleaned_weights[symbol])
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
df_buy = df_buy.append({'stock': 'CASH',
                        'momentum': 0,
                        'weights': round(1-df_buy['value'].sum()/port_value, 2),
                        'shares': 0,
                        'price': 0,
                        'value': round(port_value-df_buy['value'].sum(), 2)},
                        ignore_index=True)
df_buy = df_buy.set_index('stock')
print('Buy date', df_t.index[-1])
print(df_buy)
df_buy = df_buy.reset_index()
print(df_buy['value'].sum())

# rebalance with old portfolio
rebalance_portfolio(df_old, df_buy)
