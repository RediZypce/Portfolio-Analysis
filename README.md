# Portfolio-Analysis
Ever wondered about the perfect blend for your investment portfolio? With Python, you can optimize your investments and find the ideal company weights. Maximize returns and minimize risk with data-driven decisions!  

### Import necessary libraries and set the stock ticker symbols and date range.
    import numpy as np
    import pandas as pd
    import pandas_datareader.data as web
    import matplotlib.pyplot as plt
    import seaborn as sns
    import datetime as dt
    import yfinance as yf

    ticker = ["AAPL", "NKE", "GOOGL", 'AMZN', 'TSLA', 'NVDA', 'MNST', 'CB']
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2023, 9, 3)

### Download stock price data from Yahoo Finance.

    data = yf.download(ticker, start= "2012-01-01" , end= "2023-9-3")
    df = data['Adj Close']
    df.head()


### Calculate covariance and correlation matrices for the stocks.

    cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
    print(cov_matrix)
    corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()


### Heatmap of correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')


### Define portfolio weights and calculate portfolio variance.

    w = {'AAPL': 0.2, 'NKE': 0.2, 'GOOGL': 0.2, 'AMZN': 0.1, 'TSLA': 0.1, 'NVDA': 0.1, 'MNST': 0.05,'CB': 0.05  }
    port_var = cov_matrix.mul(w, axis=0).mul(w, axis=1).sum().sum()
    print(port_var)


### Calculate yearly returns for individual companies and the portfolio.

    ind_er = df.resample('Y').last().pct_change().mean()
    print(ind_er)
    
    w = [0.2, 0.2, 0.2, 0.1,0.1,0.1,0.05,0.05]
    port_er = (w*ind_er).sum()
    print("Portfolio Expected Return:", port_er)


###  Volatility is given by the annual standard deviation. We multiply by 250 because there are 250 trading days/year.
    
    ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))
    ann_sd


### Create a table for visualizing returns and volatility of assets.

    assets = pd.concat([ind_er, ann_sd], axis=1)
    assets.columns = ['Returns', 'Volatility']
    print(assets)


### Simulate 20,000 random portfolios and store their returns and volatility.

    p_ret = [] # Define an empty array for portfolio returns
    p_vol = [] # Define an empty array for portfolio volatility
    p_weights = [] # Define an empty array for asset weights
    
    num_assets = len(df.columns)
    num_portfolios = 20000
    for portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights/np.sum(weights)
        p_weights.append(weights)
        returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
                                          # weights 
        p_ret.append(returns)
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
        sd = np.sqrt(var) # Daily standard deviation
        ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
        p_vol.append(ann_sd)

### Dataframe of the 20000 portfolios created
    data = {'Returns':p_ret, 'Volatility':p_vol}
    
    for counter, symbol in enumerate(df.columns.tolist()):
        #print(counter, symbol)
        data[symbol+' weight'] = [w[counter] for w in p_weights]
    portfolios  = pd.DataFrame(data)
    portfolios.head()


### Plot the efficient frontier.

    sns.set(style='whitegrid')
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=portfolios, x='Volatility', y='Returns', palette='viridis', alpha=0.6)
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility')
    plt.ylabel(' Returns')


### Find and plot the minimum volatility portfolio.

    min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]   # idxmin() gives us the minimum value in the column specified.                               
    print(min_vol_port)  
    
    # plotting the minimum volatility portfolio
    plt.subplots(figsize=[5,5])
    plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
    plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=200)


### Find the optimal portfolio based on the Sharpe ratio and calculate the Sharpe ratio.

    rf = 0.03 # risk factor
    optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
    print(optimal_risky_port)

    sharpRatio= ((optimal_risky_port['Returns']-rf)/optimal_risky_port['Volatility'])
    print("Portfolio's SharpeRatio:", sharpRatio)
    print("Portfolio's Return:", optimal_risky_port['Returns'])
    print("Portfolio's Variance:", optimal_risky_port['Volatility'])
    


### Plot the optimal portfolio.

    plt.subplots(figsize=(5, 5))
    plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
    plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=200)
    plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=200)




