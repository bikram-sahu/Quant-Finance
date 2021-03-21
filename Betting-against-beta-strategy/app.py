import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
from dateutil.relativedelta import *
import copy

import streamlit as st

st.markdown("""
# Betting against beta

**AIM:** To make case for a defensive (also called betting-against-beta at times) stock strategy.

**INSTRUCTION:**

1. Using python and the data (NIFTY constituents, and their prices for the past 11 years) attached, generate monthly portfolios of top 10 stocks as per the defensive factor ranking logic, and compute the PNL.

2. Use a maximum of 12 months of lookback, so that you can have PNL performance for full 10 years. Show the performance stats you feel make sense.

3. Assume cash holding. Assume zero costs. Assume slippage free execution at closing price.

## Results:
""")

st.markdown("*Expand the plots for better visualization.*")
st.sidebar.title("Low Beta Anomaly")
n_stocks = st.sidebar.number_input('Number of least beta stocks in Portfolio?', 10)
startyear = st.sidebar.selectbox('Select start year', [2011,2012,2013,2014, 2015, 2016, 2017, 2018, 2019, 2020])
startingmonth = st.sidebar.selectbox('Select start month', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
start_month = (startyear - 2011)*12 + (startingmonth -1)

endyear = st.sidebar.selectbox('Select end year', [2011,2012,2013,2014, 2015, 2016, 2017, 2018, 2019, 2020])
endingmonth = st.sidebar.selectbox('Select end month', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
end_month = (endyear - 2011)*12 + (endingmonth - 1)

# Load data
nifty_constituents = pd.read_csv("nifty_constituents.csv", index_col = 'date')
nifty_constituents_prices = pd.read_csv("nifty_constituents_prices.csv", index_col= 'date')
nifty_50_data = pd.read_csv("NIFTY 50_Data.csv", index_col= 'Date')

nifty_constituents.index = pd.to_datetime(nifty_constituents.index)
nifty_constituents_prices.index = pd.to_datetime(nifty_constituents_prices.index)
nifty_50_data.index = pd.to_datetime(nifty_50_data.index)

# Daily returns of all the stocks
daily_returns = nifty_constituents_prices.apply(lambda row: row.pct_change(), axis=0)

nifty_50_data["ret"] = nifty_50_data.Close.pct_change()

# Extracting the month start and month end dates from the given dataframes.
df = copy.deepcopy(nifty_constituents)
df["month"] = (pd.to_datetime(df.index)).month
df["month_start"] = ~df.month.eq(df.month.shift(1))
month_start_dates = df[df.month_start == True].index.to_list()[13:]

df["month_end"] = ~df.month.eq(df.month.shift(-1))
month_end_dates = df[df.month_end == True].index.to_list()[13:]

def compute_stock_beta(stock, current_date, lookback_period=12):
    "A function to calculate beta of a stock, lookback period = 12 months"
    start_date = current_date - relativedelta(months=12)
    nifty_returns_array = np.array(nifty_50_data.loc[start_date: current_date].ret.to_list())
    stock_return_array = np.array(daily_returns.loc[start_date: current_date][stock].to_list())
    beta = np.cov(stock_return_array, nifty_returns_array)[0, 1]/np.var(nifty_returns_array)
    return beta

def top10_low_beta_stocks(on_date):
    "A function to calculate top 10 low beta stocks on a given date"
    nifty_on_date = nifty_constituents.loc[on_date]
    nifty_on_date = nifty_on_date[nifty_on_date == 1.0].index.to_list()
    nifty_stocks = pd.DataFrame(nifty_on_date, columns= {"Stocks"})
    nifty_stocks["Beta"] = nifty_stocks["Stocks"].apply(lambda row: compute_stock_beta(row, on_date))
    nifty_stocks.sort_values(by=['Beta'], inplace=True)
    
    return nifty_stocks.Stocks[:n_stocks].to_list()

def daily_return_for_the_month(portfolio, month_start, month_end):
    "function to calculate daily returns of a portfolio given a date range" 
    monthly_returns = daily_returns.loc[month_start:month_end][portfolio] +1
    monthly_returns = monthly_returns.cumprod()
    monthly_returns["temp"] = monthly_returns.apply(np.sum, axis=1)
    monthly_returns.temp = monthly_returns.temp
    monthly_returns["ret"] = (monthly_returns.temp - monthly_returns.temp.shift(1))/monthly_returns.temp.shift(1)
    monthly_returns["ret"].iloc[0] = (monthly_returns["temp"].iloc[0] - 10)/10
    monthly_returns = monthly_returns[["ret"]]
    return monthly_returns

def CAGR(df):
    CAGR = (df["cum return"].tolist()[-1])**(1/((end_month-start_month)/12)) - 1
    return CAGR

def volatility(DF):
    df = DF.copy()
    vol = df["ret"].std() * np.sqrt(len(df)/((end_month-start_month)/12))
    return vol

def sharpe(DF,rf):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df) - rf)/df["ret"].std()
    return sr
    
def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["cum_roll_max"] = df["cum return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd

portfolio_return = pd.DataFrame()
for i in range(start_month,end_month + 1,1):
    month_start = month_start_dates[i]
    month_end = month_end_dates[i]
    portfolio = top10_low_beta_stocks(month_start)
    portfolio_monthly_return = daily_return_for_the_month(portfolio, month_start, month_end)
    portfolio_return = pd.concat([portfolio_return, portfolio_monthly_return])

portfolio_return = portfolio_return
portfolio_return["cum return"] = (portfolio_return["ret"] +1).cumprod()

nifty_return = nifty_50_data.ret.loc[month_start_dates[start_month]:month_end_dates[end_month]].to_frame()
nifty_return = nifty_return
nifty_return["cum return"] = (nifty_return["ret"] +1).cumprod()



# Filter the nifty_constituents rows as per trading dates(nifty constituents prices dataframe's index)
nifty_constituents = nifty_constituents.loc[nifty_constituents.index.isin(nifty_constituents_prices.index.tolist())]

# daily returns of only nifty 50 constituent stocks
daily_returns = daily_returns * nifty_constituents

EWI = daily_returns.sum(axis=1).div(50)
EWI = EWI.to_frame(name = 'ret')
EWI = EWI.loc[month_start_dates[start_month]:month_end_dates[end_month]]

EWI["cum return"] = (EWI["ret"] +1).cumprod()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x= portfolio_return.index, y=portfolio_return["cum return"],
                    mode='lines',
                    name='Low beta portfolio'))

fig2.add_trace(go.Scatter(x= nifty_return.index, y= nifty_return["cum return"],
                    mode='lines',
                    name='MarketCap weighted Nifty50'))

fig2.add_trace(go.Scatter(x= EWI.index, y=EWI["cum return"],
                    mode='lines',
                    name='Equal Weighted Nifty50'))

fig2.update_layout(
    title="Low Beta Anomaly",
    xaxis_title="Date",
    yaxis_title="Value",
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ))

st.plotly_chart(fig2)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x= nifty_return.index, y= nifty_return["ret"],
                    mode='lines',
                    name='Nifty50'))
fig1.add_trace(go.Scatter(x= portfolio_return.index, y= portfolio_return["ret"],
                    mode='lines',
                    name='Low beta Portfolio'))
fig1.update_layout(xaxis_tickangle=-45,
                 title='Daily Returns')

st.plotly_chart(fig1)

portfolio_yearly_returns = portfolio_return["ret"] + 1

yearly_returns = nifty_return["ret"] + 1
yearly_returns = yearly_returns.resample("Y").prod() -1
yearly_returns = yearly_returns.to_frame(name="nifty_ret")
yearly_returns["portfolio_ret"] = portfolio_yearly_returns.resample("Y").prod() -1

years = yearly_returns.index.year.to_list()
fig3 = go.Figure()
fig3.add_trace(go.Bar(
    x=years,
    y=yearly_returns["nifty_ret"],
    name='Nifty50 Return',
))
fig3.add_trace(go.Bar(
    x=years,
    y=yearly_returns["portfolio_ret"],
    name='Low beta portfolio Return',
))

fig3.update_layout(barmode='group', xaxis_tickangle=-45,
                 title='Yearly Returns')

fig3.update_xaxes(
    dtick="M1",
    ticklabelmode="period")

st.plotly_chart(fig3)

nifty50 = [CAGR(nifty_return), volatility(nifty_return), sharpe(nifty_return, 0.0625), nifty_return["ret"].std(), max_dd(nifty_return)]
lowbeta= [CAGR(portfolio_return), volatility(portfolio_return), sharpe(portfolio_return, 0.0625), portfolio_return["ret"].std(), max_dd(portfolio_return)]
kpis = ["CAGR", "Volatility", "Sharpe ratio", "Std. deviation", "Max Drawdown"]
KPIs = pd.DataFrame( list(zip(nifty50, lowbeta)), index= kpis, columns=["Nifty50", "Low Beta Portfolio"])
st.sidebar.table(KPIs)

