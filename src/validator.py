import pandas as pd
import json
from backtesting.backtesting import Backtesting
from data.processor import MarketDataProcessor

def main():
    processor = MarketDataProcessor()

    stock_data, financial_data = processor.load_data()

    with open("parameter/backtesting_params.json") as file:
        backtest_params = json.load(file)

    # Load JSON file
    with open("parameter/optimized_params.json", "r", encoding="utf-8") as file:
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

    # In Sample - Optimal Parameters
    backtest = Backtesting(stock_data, financial_data, stock_score_params, quarterly_financial_score_params, yearly_financial_score_params, backtest_params.get('initial_balance'), backtest_params.get('transaction_fee'))
    results = backtest.backtest(pd.Timestamp(backtest_params.get('in_sample_start_date')), pd.Timestamp(backtest_params.get('in_sample_end_date')), save_history=True)
    with open('results/in_sample_optimized_results.json', 'w') as f:
        json.dump(results, f)

    # Out sample
    backtest = Backtesting(stock_data, financial_data, stock_score_params, quarterly_financial_score_params, yearly_financial_score_params, backtest_params.get('initial_balance'), backtest_params.get('transaction_fee'))
    results = backtest.backtest(pd.Timestamp(backtest_params.get('out_sample_start_date')), pd.Timestamp(backtest_params.get('out_sample_end_date')), save_history=True)
    with open('results/out_sample_optimized_results.json', 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()