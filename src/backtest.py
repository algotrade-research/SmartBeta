import pandas as pd
import json
from backtesting.backtesting import Backtesting
from data.processor import MarketDataProcessor

def main():
    processor = MarketDataProcessor()

    stock_data, financial_data = processor.load_data()

    with open("parameter/backtesting_params.json") as file:
        params = json.load(file)


    # Load JSON file
    with open("parameter/optimized_params.json", "r", encoding="utf-8") as file:
        params = json.load(file)

    stock_score_params = {
        'RSI': params['RSI'],
        'MACD_histogram': params['MACD_histogram'],
        # 'SMA_diff': study.best_params['SMA_diff'],
    }

    quarterly_financial_score_params = {
        'ROE': params['ROE'],
        'Net Margin': params['Net Margin'],
        'Debt to Equity': params['Debt to Equity'],
        'Current Ratio': params['Current Ratio'],
        'Asset Turnover': params['Asset Turnover'],
        'Revenue Growth': params['Revenue Growth'],
        'Quick Ratio': params['Quick Ratio'],
        'Gross Margin': params['Gross Margin'],
        'Inventory Turnover': params['Inventory Turnover'],
        # 'Debt to Asset': study.best_params['Debt to Asset'],
    }

    yearly_financial_score_params = {
        'EPS': params['EPS'],
        'PE': params['PE']
    }

    # In Sample
    backtest = Backtesting(stock_data, financial_data, stock_score_params, quarterly_financial_score_params, yearly_financial_score_params, params.get('initial_balance'), params.get('transaction_fee'))
    results = backtest.backtest(pd.Timestamp(params.get('in_sample_start_date')), pd.Timestamp('in_sample_end_date'), save_history=True)
    with open('results/in_sample_results.json', 'w') as f:
        json.dump(results, f)

    # Out sample
    backtest = Backtesting(stock_data, financial_data, stock_score_params, quarterly_financial_score_params, yearly_financial_score_params, params.get('initial_balance'), params.get('transaction_fee'))
    results = backtest.backtest(pd.Timestamp(params.get('out_sample_start_date')), pd.Timestamp('out_sample_end_date'), save_history=True)
    with open('results/in_sample_results.json', 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()