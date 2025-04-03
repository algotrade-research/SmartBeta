import optuna
import json
import pandas as pd
import os

from backtesting.backtesting import Backtesting

class ParameterOptimizer:
    def __init__(self, params_file="parameters/optimization_params.json"):
        self.params_file = params_file
        self.load_params()
        self.stock_data, self.financial_data = self.load_data()
        
    def load_params(self):
        """Load optimization parameters from JSON file"""
        try:
            with open(self.params_file, 'r') as f:
                params = json.load(f)
            
            self.start_date = pd.Timestamp(params.get('start_date', '2019-01-01'))
            self.end_date = pd.Timestamp(params.get('end_date', '2023-12-31'))
            self.n_trials = params.get('n_trials', 100)
            self.initial_balance = params.get('initial_balance', 3000000)
            self.transaction_fee = params.get('transaction_fee', 0.0035)
            
        except FileNotFoundError:
            print(f"Parameters file {self.params_file} not found. Using default values.")
            self.start_date = pd.Timestamp('2019-01-01')
            self.end_date = pd.Timestamp('2023-12-31')
            self.n_trials = 100
            self.initial_balance = 3000000
            self.transaction_fee = 0.0035
            
        except json.JSONDecodeError:
            print(f"Error parsing {self.params_file}. Using default values.")
            self.start_date = pd.Timestamp('2019-01-01')
            self.end_date = pd.Timestamp('2023-12-31')
            self.n_trials = 100
            self.initial_balance = 3000000
            self.transaction_fee = 0.0035
    
    def load_data(self):
        """Load stock and financial data"""
        # Implementation would depend on your data source
        # This is a placeholder for the data loading function
        return None, None  # Replace with actual data loading
    
    def objective(self, trial):
        """Objective function for Optuna optimization"""
        stock_score_params = {
            'RSI': trial.suggest_float('RSI', -1, 1),
            'MACD_histogram': trial.suggest_float('MACD_histogram', -1, 1),
        }

        quarterly_financial_score_params = {
            'ROE': trial.suggest_float('ROE', 0, 1),
            'Net Margin': trial.suggest_float('Net Margin', 0, 1),
            'Debt to Equity': trial.suggest_float('Debt to Equity', 0, 1),
            'Current Ratio': trial.suggest_float('Current Ratio', 0, 1),
            'Asset Turnover': trial.suggest_float('Asset Turnover', 0, 1),
            'Revenue Growth': trial.suggest_float('Revenue Growth', 0, 1),
            'Quick Ratio': trial.suggest_float('Quick Ratio', 0, 1),
            'Inventory Turnover': trial.suggest_float('Inventory Turnover', 0, 1),
        }

        yearly_financial_score_params = {
            'EPS': trial.suggest_float('EPS', 0, 1),
            'PE': trial.suggest_float('PE', 0, 1)
        }

        backtest = Backtesting(
            self.stock_data, 
            self.financial_data, 
            stock_score_params, 
            quarterly_financial_score_params, 
            yearly_financial_score_params,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_fee
        )
        
        results = backtest.backtest(
            self.start_date, 
            self.end_date, 
        )

        return results['sharpe_ratio'] + results['max_drawdown']
    
    def optimize(self):
        """Run the optimization process"""
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials)
        return study.best_params
    
    def save_results(self, best_params, output_file="parameters/optimized_params.json"):
        """Save optimization results to a file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(best_params, f, indent=4)
        
        print(f"Optimization results saved to {output_file}")


def main():
    # Create a parameter optimizer instance
    optimizer = ParameterOptimizer()
    
    # Run optimization
    best_params = optimizer.optimize()
    
    # Print and save results
    print("Best parameters found:")
    print(best_params)
    
    optimizer.save_results(best_params)


if __name__ == "__main__":
    main()