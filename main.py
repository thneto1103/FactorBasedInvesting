from data_loader import load_data
import models
from visualize import render_graphs

START_DATE = '02/2010'
END_DATE = '02/2020'

def main():
    stocks_df, index_returns = load_data()

    # Model A: Highest Return
    model1 = PortfolioModel(
        [('Mom12', False), ('PVP', True)],
        20,
        stocks_df,
        index_returns['IBOV'],
        'DU',
        'Model A: Highest Return',
        START_DATE,
        END_DATE
    )
    ModelA = model1.process()
    #render_graphs(ModelA, 'Model A: Highest Return')

    # Model B: Best Return/Volatility (Sharpe Optimized)
    model2 = PortfolioModel(
        [('Mom12', False), ('Volat', True)],
        20,
        stocks_df,
        index_returns['IBOV'],
        'GMV',
        'Model B: Return/Volatility',
        START_DATE,
        END_DATE
    )
    ModelB = model2.process()
    #render_graphs(ModelB, 'Model B: Return/Volatility')

    # Model C: Highest Alpha/Beta (Static)
    model3 = PortfolioModel(
        [('ROIC', False), ('Volat', True), ('Pat_Liq', True), ('Mom6', False)],
        20,
        stocks_df,
        index_returns['IBOV'],
        'DU',
        'Model C: Alpha/Beta',
        START_DATE,
        END_DATE
    )
    ModelC = model3.process()
    #render_graphs(ModelC, 'Model C: Alpha/Beta')

if __name__ == '__main__':
    main()
