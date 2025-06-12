import pandas as pd

def load_data():
    stocks_df = pd.read_csv('Aula-DB-Acoes.csv', index_col=0)
    index_df_raw = pd.read_csv('Aula-DB-Indices.csv')
    index_df_raw.set_index('Data', inplace=True)
    index_returns = index_df_raw.pct_change(fill_method=None)
    return stocks_df, index_returns