import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def render_graphs(dataframe, title):
    acc = dataframe[['Ret_acc', 'Ref']].copy()
    acc.columns = [title, 'Ref']
    acc.plot(figsize=(14, 4), title=title + ' - Accumulated Returns', grid=True)
    plt.show()

    ((1 + acc) / (1 + acc.shift(12)) - 1).dropna().plot.bar(figsize=(14, 4), title=title + ' - 12M Rolling Return', grid=True)
    plt.show()

    acc.rolling(60).mean().dropna().plot(figsize=(14, 4), title=title + ' - 60M SMA', grid=True)
    plt.show()

    vol = dataframe['Ret'].rolling(12).std() * np.sqrt(12) * 100
    vol.plot(figsize=(14, 4), title=title + ' - Annualized Volatility', grid=True)
    plt.show()

    dd = ((1 + acc) / (1 + acc.cummax()) - 1) * 100
    dd.plot(figsize=(14, 4), title=title + ' - Drawdown', grid=True)
    plt.show()
