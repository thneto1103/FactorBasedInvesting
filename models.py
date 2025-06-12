import pandas as pd
import numpy as np
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta as rdelta
import statsmodels.api as stats
import riskfolio as rfol


def months_between(start: str, end: str) -> float:
    delta = rdelta(dt.strptime(end, '%m/%Y'), dt.strptime(start, '%m/%Y'))
    return delta.years + delta.months / 12


class PortfolioModel:
    def __init__(self, score_fields, filter_limit, df, ref_series, strategy, label, start_date, end_date):
        self.score_fields = score_fields
        self.filter_limit = filter_limit
        self.df = df
        self.ref_series = ref_series
        self.strategy = strategy
        self.label = label
        self.cost = 0.0006
        self.result_df = None
        self.start_date = start_date
        self.end_date = end_date

    def filter_portfolio(self, date_str):
        subset = self.df.query("Data == @date_str and IBX > 0 and Retorno == Retorno").copy()
        for col, asc in self.score_fields:
            subset[f'rank_{col}'] = subset[col].rank(ascending=asc)
        subset['rank_total'] = subset[[f'rank_{col}' for col, _ in self.score_fields]].sum(axis=1).rank()
        return subset[subset['rank_total'] <= self.filter_limit].copy()

    def evaluate_static(self, selected):
        return selected['Retorno'].mean() - self.cost, selected.shape[0]

    def evaluate_optimized(self, selected, date_str):
        start_hist = dt.strptime(date_str, '%b-%Y') - rdelta(months=24)
        history = pd.date_range(start=start_hist, periods=24, freq='ME').strftime('%b-%Y')
        filt = self.df[self.df['Data'].isin(history) & self.df['Empresa'].isin(selected['Empresa'])]
        pivot = pd.pivot_table(filt, values='Retorno', index='Data', columns='Empresa')
        pivot.dropna(axis=1, thresh=6, inplace=True)

        port = rfol.Portfolio(returns=pivot.dropna())
        port.assets_stats()
        port.nea = 10
        weights = port.optimization(model='Classic', rm='MV', obj='Sharpe')

        merged = pd.merge(selected, weights, left_on='Empresa', right_index=True)
        merged['weighted_ret'] = merged['Retorno'] * merged['weights']
        ret_val = merged['weighted_ret'].sum() - self.cost
        enc = 1 / np.sum(merged['weights'] ** 2)
        return ret_val, enc

    def process(self):
        logs = []
        for cur in pd.date_range(start=self.start_date, end=self.end_date, freq='ME').strftime('%b-%Y'):
            picked = self.filter_portfolio(cur)
            if self.strategy == 'DU':
                result, enc = self.evaluate_static(picked)
            else:
                result, enc = self.evaluate_optimized(picked, cur)
            logs.append((cur, result, enc))

        log_df = pd.DataFrame(logs, columns=['Data', 'Ret', 'ENC']).set_index('Data')
        log_df['Ret_acc'] = (1 + log_df['Ret']).cumprod() - 1
        log_df['acc_max'] = log_df['Ret_acc'].cummax()
        log_df['drawdown'] = (1 + log_df['Ret_acc']) / (1 + log_df['acc_max']) - 1
        log_df['Ref'] = self.ref_series
        log_df['Ref_acc'] = (1 + log_df['Ref']).cumprod() - 1

        self.result_df = log_df
        self.display_stats()
        return log_df

    def display_stats(self):
        df = self.result_df
        ra = df['Ret_acc'].iloc[-1] * 100
        va = df['Ret'].std() * np.sqrt(12) * 100
        dd = df['drawdown'].min() * 100
        ra_ref = df['Ref_acc'].iloc[-1] * 100
        va_ref = df['Ref'].std() * np.sqrt(12) * 100

        model = stats.OLS(df['Ret'].values, stats.add_constant(df['Ref'].values), missing='drop').fit()

        print(f"\n{self.label} Portfolio")
        print(f"Ret Acc: {ra:.2f}%, Ann. Ret: {((1 + ra/100)**(1 / months_between(self.start_date, self.end_date)) - 1) * 100:.2f}%, Vol: {va:.2f}%, Drawdown: {dd:.2f}%")
        print(f"Ref Ret Acc: {ra_ref:.2f}%, Ann. Ret: {((1 + ra_ref/100)**(1 / months_between(self.start_date, self.end_date)) - 1) * 100:.2f}%, Vol: {va_ref:.2f}%")
        print(f"Alpha: {model.params[0] * 12 * 100:.2f}%, Beta: {model.params[1]:.2f}, ENC: {df['ENC'].mean():.1f}, P-values: {model.pvalues[0]:.3f} {model.pvalues[1]:.3f}\n")
