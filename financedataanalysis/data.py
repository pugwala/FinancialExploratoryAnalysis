import bs4 as bs
from collections import Counter
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pickle
import requests
from sklearn import svm, model_selection as cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from urllib.request import urlretrieve

style.use('ggplot')

# Scrape S&P 500 tickers from Wikipedia for a local reference
def save_sp500_tickers(force_download=False):
    """Get the S&P 500 tickers from Wikipedia

       Parameters
       ----------
       force_download : bool
           if True, force redownload of data

       Returns
       -------
       tickers : pandas.DataFrame
           The S&P500 tickers data

    """
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text
        # fix for . and - tickers
        mapping = str.maketrans(".","-")
        ticker = ticker.translate(mapping)
        tickers.append(ticker)

    # cache the results for local access
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    #print(tickers)
    return tickers

# Collect the ticker data using Yahoo finance API
def get_data_from_yahoo(reload_sp500=False):
    """Collect all the S&P500 transactional data from Yahoo.

    Parameters
    ----------
    reload_sp500 : bool

    Return
    ------
    tickers : list

    """
    if reload_sp500:
        tickers = pickle.load(f)
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2010,1,1)
    end = dt.datetime.now()

    for ticker in tickers:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already cached {}'.format(ticker))

#get_data_from_yahoo()

# Merge all the stock transactional data into single DataFrame
def compile_data():
    """Merge all the stock data into single DataFrame"""
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count,ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)
        df.rename(columns = {'Adj Close':ticker}, inplace=True)
        df.drop(['Open','High','Low','Close','Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')

#compile_data()


# Visualize data correlation
def visualize_data_correlation():
    """Large correlation plot"""
    df = pd.read_csv('sp500_joined_closes.csv')
    #df['AAPL'].plot()
    #plt.show()
    df_corr = df.corr()

    #print(df_corr.head())

    data = df_corr.values

    fig = plt.figure()
#    plt.rcParams['figure.figsize'] = [4,2]
    ax = fig.add_subplot(1,1,1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()

    plt.show()


