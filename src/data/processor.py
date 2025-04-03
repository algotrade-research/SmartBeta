import pandas as pd
import numpy as np
import ta
import os
from sklearn.preprocessing import StandardScaler


from data.loader import StockDataLoader  # Assuming the previous class is in this file

class MarketDataProcessor:
    def __init__(self, technical_indicators_config=None):
        """ 
        Initialize the Market Data Analyzer.
        
        :param technical_indicators_config: Configuration for technical indicators
        """
        # Default technical indicator configuration
        self.technical_indicators_config = technical_indicators_config or {
            'SMA_window': 20,
            'EMA_window': 20,
            'RSI_window': 14,
            'Bollinger_window': 20
        }
        
        # VN100 ticker list
        self.vn_100_tickers = [
            'AAA', 'ACB', 'ANV', 'BCG', 'BCM', 'BID', 'BMP', 'BSI', 'BVH', 'BWE', 
            'CII', 'CMG', 'CTD', 'CTG', 'CTR', 'CTS', 'DBC', 'DCM', 'DGC', 'DGW', 
            'DIG', 'DPM', 'DSE', 'DXG', 'DXS', 'EIB', 'EVF', 'FPT', 'FRT', 'FTS', 
            'GAS', 'GEX', 'GMD', 'GVR', 'HAG', 'HCM', 'HDB', 'HDC', 'HDG', 'HHV', 
            'HPG', 'HSG', 'HT1', 'IMP', 'KBC', 'KDC', 'KDH', 'KOS', 'LPB', 'MBB', 
            'MSB', 'MSN', 'MWG', 'NAB', 'NKG', 'NLG', 'NT2', 'OCB', 'PAN', 'PC1', 
            'PDR', 'PHR', 'PLX', 'PNJ', 'POW', 'PPC', 'PTB', 'PVD', 'PVT', 'REE', 
            'SAB', 'SBT', 'SCS', 'SHB', 'SIP', 'SJS', 'SSB', 'SSI', 'STB', 'SZC', 
            'TCB', 'TCH', 'TLG', 'TPB', 'VCB', 'VCG', 'VCI', 'VGC', 'VHM', 'VIB', 
            'VIC', 'VIX', 'VJC', 'VND', 'VNM', 'VPB', 'VPI', 'VRE', 'VTP'
        ]

    def compute_technical_indicators(self, df):
        """
        Compute technical indicators for stock price data.
        
        :param df: DataFrame with OHLCV stock data
        :return: DataFrame with added technical indicators
        """
        df = df.copy()

        # Calculate price difference
        df['close_diff'] = df['close'].diff()

        # Simple Moving Average (SMA)
        df['SMA'] = ta.trend.sma_indicator(df['close'], window=self.technical_indicators_config['SMA_window'])
        df['SMA_diff'] = df['close'] - df['SMA']

        # Exponential Moving Average (EMA)
        df['EMA'] = ta.trend.ema_indicator(df['close'], window=self.technical_indicators_config['EMA_window'])
        df['EMA_diff'] = df['close'] - df['EMA']

        # Relative Strength Index (RSI), normalized to 0-1
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=self.technical_indicators_config['RSI_window']).rsi() / 100

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            df['close'], 
            window=self.technical_indicators_config['Bollinger_window'], 
            window_dev=2
        )
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Upper_diff'] = df['close'] - df['BB_Upper']
        df['BB_Lower_diff'] = df['close'] - df['BB_Lower']

        # MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_histogram'] = macd.macd_diff()

        # Handle missing values
        fillna_map = {
            'close_diff': 0,
            'SMA': df['close'],
            'SMA_diff': 0,
            'EMA': df['close'],
            'EMA_diff': 0,
            'BB_Middle': df['close'],
            'BB_Upper': df['close'] * 1.1,
            'BB_Lower': df['close'] * 0.9,
            'BB_Upper_diff': df['close'] * -0.1,
            'BB_Lower_diff': df['close'] * 0.1,
            'MACD': 0,
            'MACD_signal': 0,
            'MACD_histogram': 0,
        }
        
        for col, fill_value in fillna_map.items():
            df[col] = df[col].fillna(fill_value)
        
        # Handle RSI separately with backward fill
        df['RSI'] = df['RSI'].bfill()
        
        return df

    def calculate_financial_ratios(self, df):
        """
        Calculate financial ratios from raw financial data.
        
        :param df: DataFrame with financial data
        :return: DataFrame with calculated financial ratios
        """
        # Sort the data for proper calculation of changes and growth rates
        df.sort_values(by=['tickersymbol', 'year', 'quarter'], inplace=True)
        
        # Define financial ratios to calculate
        financial_ratios = {
            'ROA': df['Net Profit After Tax'] / df['Total Assets'],
            'ROE': df['Net Profit After Tax'] / (df['Shareholders Equity'] + df['Owners Equity']),
            'Gross Margin': df['Gross Profit'] / df['Net Revenue'],
            'Net Margin': df['Net Profit After Tax'] / df['Net Revenue'],
            'Current Ratio': df['Current Assets'] / df['Current Liabilities'],
            'Quick Ratio': (df['Current Assets'] - df['Inventory']) / df['Current Liabilities'],
            'Debt to Equity': df['Liabilities'] / (df['Shareholders Equity'] + df['Owners Equity']),
            'Debt to Asset': df['Liabilities'] / df['Total Assets'],
            'Inventory Turnover': df['Cost of Goods Sold'] / df['Inventory'],
            'Receivable Turnover': df['Net Revenue'] / df['Current Accounts Receivable'],
            'Asset Turnover': df['Net Revenue'] / df['Total Assets'],
            'Cash Conversion Cycle': (
                df['Inventory'] * 365 / df['Cost of Goods Sold'] +
                df['Current Accounts Receivable'] * 365 / df['Net Revenue'] -
                df['Short-term Trade Payables'] * 365 / df['Cost of Goods Sold']
            ),
            'Interest Coverage Ratio': df['Operating Profit'] / df['Financial Expenses'],
            'Equity Multiplier': df['Total Assets'] / (df['Shareholders Equity'] + df['Owners Equity']),
            'Fixed Asset Turnover': df['Net Revenue'] / df['Fixed Assets'],
            'Revenue Growth': df['Net Revenue'].pct_change(),
            'Asset Growth': df['Total Assets'].pct_change(),
            'Net Income Growth': df['Net Profit After Tax'].pct_change(),
            'EPS': df['EPS']
        }
        
        # Add each ratio to the dataframe
        for ratio_name, ratio_calculation in financial_ratios.items():
            df[ratio_name] = ratio_calculation

        return df[['tickersymbol', 'year', 'quarter'] + list(financial_ratios.keys())]

    def load_data(self):
        """
        Load stock and financial data for VN100 stocks.
        
        :return: tuple of (stock_data, merged_data with financial ratios)
        """
        # Load and process stock data
        stock_data = None
        for ticker in self.vn_100_tickers:
            try:
                # Try to read CSV first
                data = pd.read_csv(f'./data/vn100/{ticker}.csv')
                data['tickersymbol'] = ticker
                data = self.compute_technical_indicators(data)
            except FileNotFoundError:
                # Initialize the stock data processor
                self.stock_data_processor = StockDataLoader()
                # If CSV not found, use StockDataLoader to generate it
                print(f"CSV for {ticker} not found. Generating using StockDataLoader...")
                data = self.stock_data_processor.get_stock_price(ticker)
                data['tickersymbol'] = ticker
                data = self.compute_technical_indicators(data)
                # Save the generated data
                os.makedirs('./data/vn100', exist_ok=True)
                data.to_csv(f'./data/vn100/{ticker}.csv', index=False)
            
            if stock_data is None:
                stock_data = data
            else:
                stock_data = pd.concat([stock_data, data], ignore_index=True)
        
        # Load financial data
        try:
            financial_data = pd.read_csv('./data/financial_report_data.csv')
        except FileNotFoundError:
            # If financial data CSV not found, use StockDataLoader to get financial data
            # Initialize the stock data processor
            self.stock_data_processor = StockDataLoader()
            print("Financial report data CSV not found. Generating using StockDataLoader...")
            financial_data = self.stock_data_processor.get_financial_statement_data()
            financial_data.to_csv('./data/financial_report_data.csv', index=False)
        
        # Filter financial data for VN100 tickers
        financial_data = financial_data[financial_data['tickersymbol'].isin(self.vn_100_tickers)]
        financial_data = self.calculate_financial_ratios(financial_data)

        # Prepare stock data for merging
        stock_data['datetime'] = pd.to_datetime(stock_data['datetime'])
        stock_data['year'] = stock_data['datetime'].dt.year

        # Find last trading day of the year
        last_trading_day = stock_data.groupby(['tickersymbol', 'year'])['datetime'].max().reset_index()

        # Calculate P/E ratio
        financial_data['PE'] = np.nan  # Initialize all with NaN first

        # Filter to only annual data (quarter == 0)
        annual_data = financial_data[financial_data['quarter'] == 0].copy()

        for index, row in annual_data.iterrows():
            # Get last trading date for this stock and year
            filtered_last_day = last_trading_day[
                (last_trading_day['tickersymbol'] == row['tickersymbol']) &
                (last_trading_day['year'] == row['year'])
            ]
            
            if filtered_last_day.empty:
                continue
                
            last_date = filtered_last_day['datetime'].iloc[0]
            
            # Get the actual stock data row with close price for this date
            stock_row = stock_data[
                (stock_data['tickersymbol'] == row['tickersymbol']) & 
                (stock_data['datetime'] == last_date)
            ]
            
            if not stock_row.empty and row['EPS'] != 0:  # Avoid division by zero
                pe_ratio = stock_row['close'].iloc[0] / row['EPS']
                financial_data.loc[index, 'PE'] = pe_ratio

        scaler = StandardScaler()

        # Exclude "tickersymbol" and "datetime"
        financial_columns_to_exclude = ["tickersymbol", "year", "quarter"]
        financial_columns_to_scale = [col for col in financial_data.columns if col not in financial_columns_to_exclude]

        stock_columns_to_exclude = ["tickersymbol", "datetime", "close"]
        stock_columns_to_scale = [col for col in stock_data.columns if col not in stock_columns_to_exclude]

        # Apply scaling only to the selected columns
        financial_data[financial_columns_to_scale] = scaler.fit_transform(financial_data[financial_columns_to_scale])
        stock_data[stock_columns_to_scale] = scaler.fit_transform(stock_data[stock_columns_to_scale])

        return stock_data, financial_data

    def close(self):
        """
        Close resources, including the stock data processor connection.
        """
        self.stock_data_processor.close_connection()