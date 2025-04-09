import pandas as pd
import json
from backtesting.backtesting import Backtesting
from data.processor import MarketDataProcessor

def main():
    processor = MarketDataProcessor()

    stock_data, financial_data = processor.load_data()

    with open("parameter/backtesting_params.json") as file:
        backtest_params = json.load(file)

    with open("parameter/initial_params.json", "r", encoding="utf-8") as file:
        weight_params = json.load(file)

    stock_score_params = {
        'RSI': weight_params['RSI'],
        'MACD_histogram': weight_params['MACD_histogram'],
    }

    quarterly_financial_score_params = {
        'ROE': weight_params['ROE'],
        'Net Margin': weight_params['Net Margin'],
        'Debt to Equity': weight_params['Debt to Equity'],
        'Current Ratio': weight_params['Current Ratio'],
        'Asset Turnover': weight_params['Asset Turnover'],
        'Revenue Growth': weight_params['Revenue Growth'],
        'Quick Ratio': weight_params['Quick Ratio'],
        'Inventory Turnover': weight_params['Inventory Turnover'],
    }

    yearly_financial_score_params = {
        'EPS': weight_params['EPS'],
        'PE': weight_params['PE']
    }

    backtest = Backtesting(stock_data, financial_data, stock_score_params, quarterly_financial_score_params, yearly_financial_score_params, backtest_params.get('initial_balance'), backtest_params.get('transaction_fee'))

    results = backtest.backtest(pd.Timestamp(backtest_params.get('in_sample_start_date')), pd.Timestamp(backtest_params.get('in_sample_end_date')))
    with open('results/in_sample_initial_results.json', 'w') as f:
        json.dump(results, f)

    backtest.plot_performance(title='In-Sample Initial Parameters', save_path='results/insample_init.png')

    print('In-sample results saved to results/in_sample_initial_results.json')
    print('Metrics:')
    print(results['metrics'])


if __name__ == "__main__":
    main()