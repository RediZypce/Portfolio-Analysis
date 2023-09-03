# Portfolio-Analysis
Max_SharpeRatio with Python 

##Import necessary libraries and set the stock ticker symbols and date range.
    import numpy as np
    import pandas as pd
    import pandas_datareader.data as web
    import matplotlib.pyplot as plt
    import datetime as dt
    import yfinance as yf

    ticker = ["AAPL", "NKE", "GOOGL", 'AMZN', 'TSLA', 'NVDA', 'MNST', 'CB']
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2023, 12, 31)

##Download stock price data from Yahoo Finance.

    data = yf.download(ticker, start=start, end=end)
    df = data['Adj Close']

##Calculate covariance and correlation matrices for the stocks.

    cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
    corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()


##Define portfolio weights and calculate portfolio variance.

    weights = {'AAPL': 0.2, 'NKE': 0.2, 'GOOGL': 0.2, 'AMZN': 0.1, 'TSLA': 0.1, 'NVDA': 0.1, 'MNST': 0.05, 'CB': 0.05}
    port_var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()


##Calculate yearly returns for individual companies and the portfolio.

    ind_er = df.resample('Y').last().pct_change().mean()
    port_er = (list(weights.values()) * ind_er).sum()


##Calculate annual portfolio volatility.

    ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))


##Create a table for visualizing returns and volatility of assets.

    assets = pd.concat([ind_er, ann_sd], axis=1)
    assets.columns = ['Returns', 'Volatility']


##Simulate 10,000 random portfolios and store their returns and volatility.

    p_ret = []
    p_vol = []
    p_weights = []
    
    num_assets = len(df.columns)
    num_portfolios = 10000
    
    for _ in range(num_portfolios):
        # Simulate portfolio weights, returns, and volatility


##Plot the efficient frontier.

    portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[5, 5])


##Find and plot the minimum volatility portfolio.

    min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
    plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)


##Find the optimal portfolio based on the Sharpe ratio and calculate the Sharpe ratio.

    rf = 0.03
    optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
    sharp_ratio = ((optimal_risky_port['Returns']-rf)/optimal_risky_port['Volatility'])


##Plot the optimal portfolio.

    plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)




