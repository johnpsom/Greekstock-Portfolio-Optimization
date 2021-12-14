# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:42:37 2021

@author: IOANNIS PSOMIADIS

ΠΡΟΣΟΧΗ ότι βλέπετε εδώ είναι φτιαγμένο για ενημερωτικούς και εκπαιδευτικούς
σκοπούς μόνο και σε καμιά περίπτωση δεν αποτελεί επενδυτική ή άλλου είδους πρόταση.
Οι επενδύσεις σε μετοχές ενέχουν οικονομικό ρίσκο και ο δημιουργός της εφαρμογής
δεν φέρει καμιά ευθύνη σε περίπτωση απώλειας περιουσίας.
Μπορείτε να επικοινωνείτε τα σχόλια και παρατηρήσεις σας
στο email: getyour.portfolio@gmail.com .

Υπολογισμός βέλτιστου χαρτοφυλακίου από 100+ επιλεγμένες μετοχές του ΧΑ,
βασισμένος στις αρχές της Σύγχρονης Θεωρίας Χαρτοφυλακίου.
Στόχος είναι να αυτοματοποιηθεί όλη η διαδικασία με βάση την στρατηγική που
περιγράφεται παρακάτω.

Η εφαρμογή χρησιμοποιεί ιστορικές τιμές για όλες τις μετοχές της παραπάνω λίστας.
Οι επιπλεόν παράμετροι της στρατηγικής μας είναι
- ένας τεχνικός δείκτης momentum για να βρίσκει ποιές από αυτές έχουν δυναμική για άνοδο της τιμής τους
- το μέγιστο πλήθος μετοχών που θέλουμε να έχουμε στο χαρτοφυλάκιό μας (π.χ. 5, 10 ΄ή 15 μετοχές)
- το ελάχιστο ποσοστό συμμετοχής της κάθε μετοχής στο επιλεγμένο χαρτοφυλάκιο. (π.χ 5% ή 10%)
- το χρονικό διάστημα διακράτησης του προτεινόμενου χαρτοφυλακίου σε ημέρες (π.χ. 5, 10 ή 20 μέρες)

Η στρατηγική μας είναι αφού δοκιμάσουμε όλους τους συνδυασμούς των παραπάνω
παραμέτρων στο παρελθόν με χρήση των ιστορικών τιμών όλων των μετοχών να επιλέγουμε
κάθε φορά τον καλύτερο συνδυασμό και μετά να δημιουργούμε το χαρτοφυλάκιό μας,
ελπίζοντας ότι η δυναμική αυτή θα είναι σε ισχύ για κάποιο χρονικό διάστημα ακόμη.
"""

import warnings
import pandas as pd
import greekstocks
from stock_tickers import tickers_gr

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")

#get all greek stock tickers OHLC price data
greekstocks_data=greekstocks.get_greekstocks_data(tickers_gr)
#create the close prices dataframe and other data
l_close=pd.DataFrame(columns=['stock','date','lastprice','len_prices'])
close_data=pd.DataFrame()
i=1
for ticker in tickers_gr:
    last_close=greekstocks_data[ticker].iloc[-1]['close']
    last_date=greekstocks_data[ticker].iloc[-1]['date']
    len_values=len(greekstocks_data[ticker])
    l_close=l_close.append({'stock':ticker,'date':last_date,'lastprice':last_close,
                            'len_prices':len_values},ignore_index=True)
    df_temp=greekstocks_data[ticker].loc[:,['date','close']].rename(columns={'close':ticker}) \
                                                            .set_index('date')
    if i==1:
        close_data=df_temp
        i=i+1
    else:
        close_data=close_data.merge(df_temp,how='inner',on='date')

l_close_min=l_close['len_prices'].min()


#Declare Constants
portfolio_value=10000
new_portfolio_value=0
df=close_data
#momentum_window = v
#minimum_momentum = 70
#portfolio_size=15
#cutoff=0.1
#added_value how much cash to add each trading period
#tr_period=21 #trading period, 21 is a month,10 in a fortnite, 5 is a week, 1 is everyday
#dataset=800 #start for length of days used for the optimising dataset
#l_days=600  #how many days to use in optimisations
backtest_results=pd.DataFrame(columns=['trades', 'momentum_window',
                                       'minimum_momentum', 'portfolio_size',
                                       'tr_period', 'cutoff',
                                       'tot_contribution', 'final portfolio_value',
                                       'cumprod', 'tot_ret', 'drawdown'])
#backtest 80%
x=int(0.9*l_close_min)
df_bt=df.head(x) #backtest dataframe of first x values from total prices
df_vld=df.tail(x) #validate dataframe of the rest prices like a forward test
#run all the combinations for all parameter values
for backtest_days in [100]:
    for momentum_window in range(200,510,50):
        for minimum_momentum in range(90,210,30):
            for portfolio_size in [5,10,15,20]:
                for trading_period in [5,10,20]:
                    for cutoff in [0.01,0.05,0.1]:
                        portfolio_value=10000
                        backtest_dataset=len(df_bt)-backtest_days
                        lookback_days=momentum_window
                        added_value=0
                        bt_result=greekstocks.backtest_portfolio(df_bt,backtest_dataset,
                                                                 lookback_days,momentum_window,
                                                                 minimum_momentum,portfolio_size,
                                                                 trading_period,cutoff,
                                                                 portfolio_value,added_value)
                        backtest_results=backtest_results.append(bt_result, ignore_index=True)
                        print(bt_result)
                        #plt.plot(plotted_portval)
                        print(backtest_results.sort_values(by=['tot_ret']).tail(2))
print(backtest_results.sort_values(by=['tot_ret']).tail(10))

best_backtest_result=backtest_results.sort_values(by=['tot_ret']).tail(1).reset_index(drop=True)
print(best_backtest_result)


#validate the best of the best portfolio
portfolio_value=10000
momentum_window = int(best_backtest_result.loc[0,'momentum_window'])
minimum_momentum = int(best_backtest_result.loc[0,'minimum_momentum'])
portfolio_size=int(best_backtest_result.loc[0,'portfolio_size'])
cutoff=int(best_backtest_result.loc[0,'cutoff'])
trading_period=int(best_backtest_result.loc[0,'tr_period'])
validation_dataset=len(df_vld)-backtest_days
lookback_days=momentum_window
added_value=0
bt_result=greekstocks.backtest_portfolio2(df_vld,validation_dataset,
                                          lookback_days,momentum_window,
                                          minimum_momentum,portfolio_size,
                                          trading_period,cutoff,
                                          portfolio_value,added_value)
print(bt_result)


#create final and new portfolio
df_old=pd.read_csv('greekstocks_portfolio.csv').iloc[:,1:]
cash=df_old.loc[df_old['stock']=='CASH']['value'].values[0]
latest_prices = greekstocks.get_latest_prices(df)
new_price=[]
for symbol in df_old['stock'][:-1].values.tolist():
    new_price.append(latest_prices[symbol])
new_price.append(cash)
df_old['new_prices']=new_price
df_old['new_value']=df_old['new_prices']*df_old['shares']
portfolio_value=0.9*(cash+df_old['new_value'].sum())
#portfolio_value=2200
df_tr=close_data.tail(momentum_window)
df_m=pd.DataFrame()
m_s=[]
st=[]
for s in tickers_gr:
    st.append(s)
    m_s.append(greekstocks.momentum_score(df_tr[s].tail(momentum_window)))
df_m['stock']=st
df_m['momentum']=m_s
dev=df_m['momentum'].std()
# Get the top momentum stocks for the period
df_m = df_m.sort_values(by='momentum', ascending=False)
df_m=df_m[(df_m['momentum']>minimum_momentum-0.5*dev)&(df_m['momentum']<minimum_momentum+1.9*dev)].head(portfolio_size)
# Set the universe to the top momentum stocks for the period
universe = df_m['stock'].tolist()
print(universe)
# Create a df with just the stocks from the universe
df_buy= greekstocks.get_portfolio(universe, df_tr, portfolio_value, cutoff, df_m)[0]
df_buy=df_buy.reset_index()
print(df_buy)
print(df_buy['value'].sum())

#rebalance with old portfolio
greekstocks.rebalance_portfolio(df_old,df_buy)
