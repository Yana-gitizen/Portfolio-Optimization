#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.optimize as sco


# # Define the stock tickers

# In[ ]:


tickers = ['CALX', 'NOVT', 'RGEN', 'LLY',
'AMD', 'NFLX', 'COST', 'BJ', 'WING',
'MSCI', 'CBRE']  # Example tickers

# Download historical data
data = yf.download(tickers, start='2020-01-01', end='2025-01-01')['Close']


# # Calculate Daily Returns, Mean Returns and Covariance Matrix

# In[27]:


returns = data.pct_change().dropna()

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()


# # Define Portfolio Performance Functions based on Weights

# In[17]:


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252  # Annualized return
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized volatility
    return std, returns


# # Define Optimization Functions based on Sharpe Ratio

# In[19]:


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
    bounds = tuple((0, 1) for asset in range(num_assets))  # No short selling
    result = sco.minimize(neg_sharpe_ratio, num_assets * [1. / num_assets,], args=args,
                           method='SLSQP', bounds=bounds, constraints=constraints)
    return result


# # Simulate Portfolios and Plot Efficient Frontier

# In[23]:


def simulate_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)  # Normalize to sum to 1
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev  # Sharpe ratio
    
    return results, weights_record

# Parameters
num_portfolios = 10000
risk_free_rate = 0.0178  # Example risk-free rate

# Simulate portfolios
results, weights = simulate_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

# Plotting
plt.figure(figsize=(10, 7))
plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Returns')
plt.title('Efficient Frontier')
plt.show()


# # Find and Plot Maximum Sharpe Ratio Portfolio

# In[25]:


max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)

plt.scatter(sdp, rp, marker='*', color='r', s=500, label='Maximum Sharpe Ratio')
plt.legend(labelspacing=0.8)
plt.show()


# # Annualized Returns and Volatility for Each Stock

# In[29]:


# Calculate mean returns and covariance matrix
mean_returns = returns.mean() * 252  # Annualized mean returns
cov_matrix = returns.cov() * 252  # Annualized covariance matrix


# In[31]:


# Calculate annualized volatility for each stock
annualized_volatility = returns.std() * np.sqrt(252)

# Create a DataFrame to store the results
metrics = pd.DataFrame({
    'Annualized Return': mean_returns,
    'Annualized Volatility': annualized_volatility})

# Display the metrics
print(metrics)

# Optional: Plot the return vs volatility for each stock
plt.figure(figsize=(10, 6))
plt.scatter(metrics['Annualized Volatility'], metrics['Annualized Return'], marker='o')

for i, txt in enumerate(metrics.index):
    plt.annotate(txt, (metrics['Annualized Volatility'][i], metrics['Annualized Return'][i]))

plt.title('Return vs Volatility for Each Stock')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.grid()
plt.show()


# # Entire Portfolio Return and Volatility

# In[33]:


# Define portfolio weights (example: equal weights)
weights = np.array([1/len(tickers)] * len(tickers))  # Equal weights for simplicity

# Calculate total portfolio annualized return
portfolio_return = np.sum(weights * metrics['Annualized Return'])

# Calculate total portfolio annualized volatility
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Display the results
print(f'Total Portfolio Annualized Return: {portfolio_return:.4f}')
print(f'Total Portfolio Annualized Volatility: {portfolio_volatility:.4f}')


# In[ ]:




