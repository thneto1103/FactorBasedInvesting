import riskfolio as rp
import numpy as np
import pandas as pd
from datetime import datetime
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# ---- Config ----
start = '02/2010'
end = '02/2020'

# ---- Load Data ----
dados = pd.read_csv('Aula-DB-Acoes.csv', index_col=0)
indices_acc = pd.read_csv('Aula-DB-Indices.csv')
indices_acc.set_index('Data', inplace=True)
indices = indices_acc.pct_change(fill_method=None)

# ---- Date Utility ----
def calc_dif_dates(start, end):
    d = relativedelta(datetime.strptime(end, '%m/%Y'), datetime.strptime(start, '%m/%Y'))
    return d.years + d.months / 12

# ---- 2-Factor Model ----
def create_port_2_fatores(f1, a1, f2, a2, filtro_fim, dados, referencia, otim, name):
    cost_trans = 0.0006
    list_date, list_ret = [], []
    min_enc = 100

    for dt in pd.date_range(start=start, end=end, freq='ME').strftime('%b-%Y'):
        cart = dados[(dados['Data'] == dt) & (dados['IBX'] > 0) & (dados['Retorno'].notnull())].copy()
        cart['rank1'] = cart[f1].rank(ascending=a1)
        cart['rank2'] = cart[f2].rank(ascending=a2)
        cart['rank_sum'] = cart['rank1'] + cart['rank2']
        cart['rank_total'] = cart['rank_sum'].rank(ascending=True)
        port_temp = cart[cart['rank_total'] <= filtro_fim].copy()

        if otim == 'DU':
            ret = port_temp['Retorno'].mean() - cost_trans
            min_enc = filtro_fim
        else:
            hist_start = datetime.strftime(datetime.strptime(dt, '%b-%Y') - relativedelta(months=24), '%b-%Y')
            filter_dates = pd.date_range(start=hist_start, periods=24, freq='ME').strftime('%b-%Y')
            dados_melt = dados[(dados['Data'].isin(filter_dates)) & (dados['Empresa'].isin(port_temp['Empresa']))]
            dados_pivot = pd.pivot_table(dados_melt, values='Retorno', index='Data', columns='Empresa')
            dados_pivot.dropna(axis=1, thresh=6, inplace=True)

            port = rp.Portfolio(returns=dados_pivot.dropna())
            port.assets_stats()
            port.nea = 10
            weights = port.optimization(model='Classic', rm='MV', obj='MinRisk')
            merged = pd.merge(port_temp, weights, left_on='Empresa', right_index=True)
            merged['Ret_w'] = merged['Retorno'] * merged['weights']
            ret = merged['Ret_w'].sum() - cost_trans
            min_enc = 1 / np.sum(merged['weights']**2)

        list_date.append(dt)
        list_ret.append(ret)

    Port = pd.DataFrame({'Data': list_date, 'Ret': list_ret}).set_index('Data')
    Port['Ret_acc'] = (1 + Port['Ret']).cumprod() - 1
    Port['acc_max'] = Port['Ret_acc'].cummax()
    Port['drawdown'] = ((1 + Port['Ret_acc']) / (1 + Port['acc_max'])) - 1
    Port['Ref'] = referencia
    Port['Ref_acc'] = (1 + Port['Ref']).cumprod() - 1

    # ---- Metrics ----
    ret_acc = Port['Ret_acc'].iloc[-1]*100
    vol_aa = Port['Ret'].std() * (12 ** 0.5) * 100
    drawdown = Port['drawdown'].min()*100
    ret_ref_acc = Port['Ref_acc'].iloc[-1]*100
    vol_ref_aa = Port['Ref'].std() * (12 ** 0.5) * 100

    model = sm.OLS(Port['Ret'].values, sm.add_constant(Port['Ref'].values), missing='drop').fit()

    print(f"\n{name} Portfolio [{f1}, {f2}, {filtro_fim}, {otim}]")
    print(f"Port Ret Acc: {ret_acc:.2f}%  Ret anual.: {(pow(ret_acc/100+1, 1/calc_dif_dates(start, end))-1)*100:.2f}%  Vol anual.: {vol_aa:.2f}% Drawdown: {drawdown:.2f}%")
    print(f"Ref  Ret Acc: {ret_ref_acc:.2f}%  Ret anual.: {(pow(ret_ref_acc/100+1, 1/calc_dif_dates(start, end))-1)*100:.2f}%  Vol anual.: {vol_ref_aa:.2f}%")
    print(f"Port Alpha: {model.params[0]*12*100:.2f}% Beta: {model.params[1]:.2f} ENC: {min_enc:.1f}  / P-values: {model.pvalues[0]:.3f} {model.pvalues[1]:.3f}\n")

    return Port

# ---- 4-Factor Model ----
def create_port_4_fatores_par(f1, a1, f2, a2, f3, a3, f4, a4, filtro_fim, dados, referencia, otim, name):
    cost_trans = 0.0006
    list_date, list_ret = [], []
    min_enc = 100

    for dt in pd.date_range(start=start, end=end, freq='ME').strftime('%b-%Y'):
        cart = dados[(dados['Data'] == dt) & (dados['IBX'] > 0) & (dados['Retorno'].notnull())].copy()
        cart['rank1'] = cart[f1].rank(ascending=a1)
        cart['rank2'] = cart[f2].rank(ascending=a2)
        cart['rank3'] = cart[f3].rank(ascending=a3)
        cart['rank4'] = cart[f4].rank(ascending=a4)
        cart['rank_sum'] = cart['rank1'] + cart['rank2'] + cart['rank3'] + cart['rank4']
        cart['rank_total'] = cart['rank_sum'].rank(ascending=True)
        port_temp = cart[cart['rank_total'] <= filtro_fim].copy()

        if otim == 'DU':
            ret = port_temp['Retorno'].mean() - cost_trans
            min_enc = filtro_fim
        else:
            ret = port_temp['Retorno'].mean() - cost_trans
            min_enc = port_temp.shape[0]

        list_date.append(dt)
        list_ret.append(ret)

    Port = pd.DataFrame({'Data': list_date, 'Ret': list_ret}).set_index('Data')
    Port['Ret_acc'] = (1 + Port['Ret']).cumprod() - 1
    Port['acc_max'] = Port['Ret_acc'].cummax()
    Port['drawdown'] = ((1 + Port['Ret_acc']) / (1 + Port['acc_max'])) - 1
    Port['Ref'] = referencia
    Port['Ref_acc'] = (1 + Port['Ref']).cumprod() - 1

    # ---- Metrics ----
    ret_acc = Port['Ret_acc'].iloc[-1]*100
    vol_aa = Port['Ret'].std() * (12 ** 0.5) * 100
    drawdown = Port['drawdown'].min()*100
    ret_ref_acc = Port['Ref_acc'].iloc[-1]*100
    vol_ref_aa = Port['Ref'].std() * (12 ** 0.5) * 100

    model = sm.OLS(Port['Ret'].values, sm.add_constant(Port['Ref'].values), missing='drop').fit()

    print(f"\n{name} Portfolio [{f1}, {f2}, {f3}, {f4}, {filtro_fim}, {otim}]")
    print(f"Port Ret Acc: {ret_acc:.2f}%  Ret anual.: {(pow(ret_acc/100+1, 1/calc_dif_dates(start, end))-1)*100:.2f}%  Vol anual.: {vol_aa:.2f}% Drawdown: {drawdown:.2f}%")
    print(f"Ref  Ret Acc: {ret_ref_acc:.2f}%  Ret anual.: {(pow(ret_ref_acc/100+1, 1/calc_dif_dates(start, end))-1)*100:.2f}%  Vol anual.: {vol_ref_aa:.2f}%")
    print(f"Port Alpha: {model.params[0]*12*100:.2f}% Beta: {model.params[1]:.2f} ENC: {min_enc:.1f}  / P-values: {model.pvalues[0]:.3f} {model.pvalues[1]:.3f}\n")

    return Port

# ---- Plotting ----
def plot_results(Port, title):
    acc = pd.DataFrame(Port['Ret_acc'])
    acc['Ref'] = Port['Ref_acc']
    acc.columns = [title, 'Ref']
    acc.plot(figsize=(14, 4), title=title + " - Cumulative Return", grid=True)
    plt.show()

    ((1 + acc) / (1 + acc.shift(12)) - 1).dropna()\
        .plot.bar(figsize=(14, 4), title=title + " - Rolling 12M Returns", grid=True)
    plt.show()

    acc.rolling(60).mean().dropna().plot(figsize=(14, 4), title=title + " - 60M Moving Average", grid=True)
    plt.show()

    vol_series = pd.Series(Port['Ret']).rolling(12).std() * (12 ** 0.5) * 100
    vol_series.plot(figsize=(14, 4), title=title + " - Annualized Volatility", grid=True)
    plt.show()

    drawdown = ((1 + acc) / (1 + acc.cummax()) - 1) * 100
    drawdown.plot(figsize=(14, 4), title=title + " - Drawdown", grid=True)
    plt.show()

# ---- Run Portfolios ----

# Model 1: Highest Return
Port1 = create_port_2_fatores('Mom12', False, 'PVP', True, 20, dados, indices['IBOV'], 'GMV', 'Model 1: Highest Return')
plot_results(Port1, 'Model 1: Highest Return')

# Model 2: Return/Volatility
Port2 = create_port_2_fatores('Mom12', False, 'Volat', True, 20, dados, indices['IBOV'], 'GMV', 'Model 2: Return/Volatility')
plot_results(Port2, 'Model 2: Return/Volatility')

# Model 3: Alpha/Beta
Port3 = create_port_4_fatores_par('ROIC', False, 'Volat', True, 'Pat_Liq', True, 'Mom6', False, 20, dados, indices['IBOV'], 'DU', 'Model 3: Alpha/Beta')
plot_results(Port3, 'Model 3: Alpha/Beta')
