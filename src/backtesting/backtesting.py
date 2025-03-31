import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class Backtesting:
    def __init__(self, stock_data, financial_data, stock_score_params, quarterly_financial_score_params, yearly_financial_score_params, initial_balance=3000000, transaction_cost=0.0035):
        self.stock_data = stock_data
        self.financial_data = financial_data
        self.stock_score_params = stock_score_params
        self.quarterly_financial_score_params = quarterly_financial_score_params
        self.yearly_financial_score_params = yearly_financial_score_params
        self.balance = initial_balance
        self.transaction_cost = transaction_cost
        self.portfolio = {}  # Format: {tickersymbol: {'shares': shares, 'buy_date': date, 'buy_price': price}}
        self.history = []  # To track daily portfolio and balance
        self.stock_pnl_history = []  # To track daily P&L for each stock
        
        # Create indexes for faster lookups
        self._create_indexes()
        
        # Cache for stock prices and scores
        self.price_cache = {}
        self.score_cache = {}
    
    def _create_indexes(self):
        # Create dictionary lookups instead of filtering dataframes repeatedly
        self.stock_price_lookup = {}
        for _, row in self.stock_data.iterrows():
            ticker = row['tickersymbol']
            date = row['datetime']
            key = (ticker, date)
            self.stock_price_lookup[key] = row
            
        # Create financial data lookups
        self.yearly_financial_lookup = {}
        self.quarterly_financial_lookup = {}
        
        for _, row in self.financial_data.iterrows():
            ticker = row['tickersymbol']
            year = row['year']
            quarter = row['quarter']
            
            if quarter == 0:  # Yearly data
                self.yearly_financial_lookup[(ticker, year)] = row
            else:  # Quarterly data
                self.quarterly_financial_lookup[(ticker, year, quarter)] = row
    
    def calculate_score(self, tickersymbol, datetime):
        # Check cache first
        cache_key = (tickersymbol, datetime)
        if cache_key in self.score_cache:
            return self.score_cache[cache_key]
            
        stock_score = 0
        financial_score = 0
        
        # Calculate stock score
        stock_key = (tickersymbol, datetime)
        if stock_key in self.stock_price_lookup:
            stock_slice = self.stock_price_lookup[stock_key]
            for key, value in self.stock_score_params.items():
                if key in stock_slice:
                    score_component = stock_slice.get(key, 0) * value
                    if not pd.isna(score_component):  # Check if not NaN
                        stock_score += score_component

        # Calculate yearly financial score
        yearly_key = (tickersymbol, datetime.year - 1)
        if yearly_key in self.yearly_financial_lookup:
            financial_slice = self.yearly_financial_lookup[yearly_key]
            for key, value in self.yearly_financial_score_params.items():
                if key in financial_slice:
                    score_component = financial_slice.get(key, 0) * value
                    if not pd.isna(score_component):  # Check if not NaN
                        financial_score += score_component
        
        # Calculate quarterly financial score
        if datetime.month < 4:
            quarter = 4
            year = datetime.year - 1
        else:
            quarter = (datetime.month - 1) // 3 + 1
            year = datetime.year
        
        quarterly_key = (tickersymbol, year, quarter)
        if quarterly_key in self.quarterly_financial_lookup:
            financial_slice = self.quarterly_financial_lookup[quarterly_key]
            for key, value in self.quarterly_financial_score_params.items():
                if key in financial_slice:
                    score_component = financial_slice.get(key, 0) * value
                    if not pd.isna(score_component):  # Check if not NaN
                        financial_score += score_component
        
        total_score = stock_score + financial_score
        
        # Handle the case where the total score might still be NaN
        if pd.isna(total_score):
            total_score = 0  # or some default value
            
        self.score_cache[cache_key] = total_score
        return total_score

    def get_stock_price(self, tickersymbol, datetime):
        # Check price cache first
        cache_key = (tickersymbol, datetime)
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        
        if cache_key in self.stock_price_lookup:
            price = self.stock_price_lookup[cache_key]['close']
            self.price_cache[cache_key] = price
            return price
        
        return None

    def buy_stock(self, tickersymbol, dt, amount, score):
        price = self.get_stock_price(tickersymbol, dt)
        if price is None:
            return False
        
        # Calculate shares to buy with transaction cost
        shares = int(amount / (price * (1 + self.transaction_cost)))
        # Round down to nearest multiple of 100
        shares = (shares // 100) * 100
        if shares <= 0:
            return False
        
        total_cost = shares * price * (1 + self.transaction_cost)
        if total_cost > self.balance:
            return False
        
        buy_date = dt
        buy_date_str = buy_date.strftime("%Y-%m-%d %H:%M:%S") if isinstance(buy_date, datetime) else str(buy_date)
        
        # Update portfolio and balance
        if tickersymbol in self.portfolio:
            # Average down/up if we already have this stock
            existing_shares = self.portfolio[tickersymbol]['shares']
            existing_cost = existing_shares * self.portfolio[tickersymbol]['buy_price']
            total_shares = existing_shares + shares
            avg_price = (existing_cost + (shares * price)) / total_shares
            
            self.portfolio[tickersymbol] = {
                'shares': total_shares,
                'buy_date': buy_date_str,
                'buy_price': avg_price,
                'score': score,
                'max_price': max(self.portfolio[tickersymbol]['max_price'], price),
                'max_price_date': buy_date_str
            }
        else:
            self.portfolio[tickersymbol] = {
                'shares': shares,
                'buy_date': buy_date_str,
                'buy_price': price,
                'score': score,
                'max_price': price,
                'max_price_date': buy_date_str
            }
            
        self.balance -= total_cost
        return True

    def sell_stock(self, tickersymbol, sell_datetime):
        if tickersymbol not in self.portfolio or self.portfolio[tickersymbol]['shares'] <= 0:
            return False
        
        price = self.get_stock_price(tickersymbol, sell_datetime)
        if price is None:
            return False
        
        shares = self.portfolio[tickersymbol]['shares']
        sell_value = shares * price * (1 - self.transaction_cost)

        # Convert buy_date and sell_date to strings
        buy_date = self.portfolio[tickersymbol]['buy_date']
        buy_date_str = buy_date.strftime("%Y-%m-%d %H:%M:%S") if isinstance(buy_date, datetime) else str(buy_date)
        sell_date_str = sell_datetime.strftime("%Y-%m-%d %H:%M:%S") if isinstance(sell_datetime, datetime) else str(sell_datetime)

        # add to stock_pnl_history
        self.stock_pnl_history.append({
            'tickersymbol': tickersymbol,
            'score': self.portfolio[tickersymbol]['score'],
            'buy_date': buy_date_str,
            'buy_price': self.portfolio[tickersymbol]['buy_price'],
            'sell_date': sell_date_str,
            'sell_price': price,
            'shares': shares,
            'pnl': sell_value - (shares * self.portfolio[tickersymbol]['buy_price'])
        })
        
        self.balance += sell_value
        del self.portfolio[tickersymbol]
        return True

    def calculate_portfolio_value(self, dt):
        date = dt.strftime("%Y-%m-%d %H:%M:%S") if isinstance(dt, datetime) else str(dt)
        portfolio_value = 0
        for ticker, data in self.portfolio.items():
            price = self.get_stock_price(ticker, dt)
            if price is not None:
                portfolio_value += data['shares'] * price
                if price > data['max_price']:
                    self.portfolio[ticker]['max_price'] = price
                    self.portfolio[ticker]['max_price_date'] = date
        return portfolio_value

    def save_daily_status(self, dt):
        portfolio_value = self.calculate_portfolio_value(dt)
        total_value = self.balance + portfolio_value

        date = dt.strftime("%Y-%m-%d %H:%M:%S") if isinstance(dt, datetime) else str(dt)

        daily_return = None
        if len(self.history) > 0:
            previous_total = self.history[-1]['total_value']
            daily_return = (total_value - previous_total) / previous_total if previous_total > 0 else 0
        
        daily_status = {
            'date': date,
            'balance': self.balance,
            'portfolio_value': portfolio_value,
            'total_value': total_value,
            'daily_return': daily_return,
            'portfolio': {ticker: {**data} for ticker, data in self.portfolio.items()}
        }
        
        self.history.append(daily_status)
        return daily_status

    def backtest(self, start_date, end_date, save_history=False, risk_free_rate=0.03/252):
        # Pre-calculate trading days and unique tickers for faster processing
        trading_days = sorted(self.stock_data[
            (self.stock_data['datetime'] >= start_date) &
            (self.stock_data['datetime'] <= end_date)
        ]['datetime'].unique())
        
        all_tickers = set(self.stock_data['tickersymbol'].unique())
        
        # Pre-compute date differences to avoid calculating them repeatedly
        date_diffs = {}
        
        # Calculate initial values
        if save_history:
            initial_status = self.save_daily_status(trading_days[0])
            initial_balance = initial_status['balance']
        else:
            initial_balance = self.balance
        
        for current_date in trading_days:
            # Check for stocks to sell (bought 3 months ago)
            stocks_to_sell = []
            for ticker, data in self.portfolio.items():
                buy_date = data['buy_date']
                buy_date_ts = pd.Timestamp(buy_date)
                date_diff_key = (buy_date_ts, current_date)
                
                if date_diff_key not in date_diffs:
                    date_diffs[date_diff_key] = (current_date - buy_date_ts).days
                
                if date_diffs[date_diff_key] >= 90:
                    stocks_to_sell.append(ticker)
            
            for ticker in stocks_to_sell:
                self.sell_stock(ticker, current_date)
            
            # Avoid calculating scores for tickers we already own
            owned_tickers = set(self.portfolio.keys())
            tickers_to_evaluate = all_tickers - owned_tickers
            
            # Calculate scores for available tickers on this day
            scores = []
            for ticker in tickers_to_evaluate:
                try:
                    score = self.calculate_score(ticker, current_date)
                    # Don't add tickers with no price data
                    if self.get_stock_price(ticker, current_date) is not None:
                        scores.append((ticker, score))
                except:
                    continue
            
            # Sort by score and get top 3
            scores.sort(key=lambda x: x[1], reverse=True)
            top_tickers = scores[:3]
            
            # Buy top 3 stocks (divide remaining balance)
            max_per_stock = min(300000, self.balance / 3)
            if max_per_stock > 0:
                for ticker, score in top_tickers:
                    if ticker not in self.portfolio:  # Only buy if we don't already own it
                        self.buy_stock(ticker, current_date, max_per_stock, score)
            
            # Save daily status
            if save_history:
                self.save_daily_status(current_date)
                    
        # Calculate final performance
        final_portfolio_value = self.calculate_portfolio_value(trading_days[-1])
        total_final_value = self.balance + final_portfolio_value

        # Calculate Sharpe ratio if we have history
        sharpe_ratio = None
        if save_history and len(self.history) > 1:
            # Extract daily returns, skipping the first day which has no return
            daily_returns = [record['daily_return'] for record in self.history if record['daily_return'] is not None]
            
            if daily_returns:
                # Calculate average daily return and standard deviation
                avg_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                
                # Calculate annualized Sharpe ratio (assuming 252 trading days per year)
                if std_return > 0:
                    daily_sharpe = (avg_return - risk_free_rate) / std_return
                    sharpe_ratio = daily_sharpe * np.sqrt(252)  # Annualize
        
        # Calculate max drawdown
        max_value = initial_balance
        max_drawdown = 0
        for record in self.history:
            total_value = record['total_value']
            max_value = max(max_value, total_value)
            drawdown = (total_value - max_value) / max_value
            max_drawdown = min(max_drawdown, drawdown)
        
        return {
            'initial_value': initial_balance,
            'final_value': total_final_value,
            'profit_loss': total_final_value - initial_balance,
            'profit_loss_percent': ((total_final_value / initial_balance) - 1) * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'history': self.history,
            'stock_pnl_history': self.stock_pnl_history
        }

    def plot_performance(self):        
        if not self.history:
            print("No backtest history to plot")
            return
        
        dates = [record['date'] for record in self.history]
        balance = [record['balance'] for record in self.history]
        portfolio_value = [record['portfolio_value'] for record in self.history]
        total_value = [record['total_value'] for record in self.history]
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'Date': dates,
            'Cash Balance': balance,
            'Portfolio Value': portfolio_value,
            'Total Value': total_value
        })

        # Convert date strings to datetime objects
        df['Date'] = pd.to_datetime(df['Date'])

        # Load VN-Index data for comparison
        vn_index = pd.read_csv('./vn100/VNINDEX.csv')
        vn_index['datetime'] = pd.to_datetime(vn_index['datetime'])
        vn_index = vn_index[vn_index['datetime'] >= df['Date'].min()]
        vn_index = vn_index[vn_index['datetime'] <= df['Date'].max()]

        # Suppose that we invest all money in VN-Index at the beginning
        vn_index['Total Value'] = balance[0] * (vn_index['close'] / vn_index['close'].iloc[0])
        
        # Plot the results
        plt.figure(figsize=(12, 8))
        plt.plot(df['Date'], df['Total Value'], label='Total Value')
        # plt.plot(df['Date'], df['Portfolio Value'], label='Portfolio Value')
        # plt.plot(df['Date'], df['Cash Balance'], label='Cash Balance')

        plt.plot(vn_index['datetime'], vn_index['Total Value'], label='VN-Index', linestyle='--', color='orange')


        
        plt.title('Backtest Performance')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return df