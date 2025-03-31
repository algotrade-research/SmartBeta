import psycopg2
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime

from config.config import db_params

class StockDataLoader:
    def __init__(self, connection_params=None):
        """
        Initialize the StockDataProcessor with database connection parameters.
        
        :param connection_params: Dictionary of database connection parameters
        """
        if connection_params is None:
            connection_params = db_params
        
        self.connection = psycopg2.connect(**connection_params)
        self.vn_100_tickers = [
            'AAA', 'ACB', 'ANV', 'BCG', 'BCM', 'BID', 'BMP', 'BSI', 'BVH', 'BWE', 'CII', 'CMG', 'CTD', 'CTG', 'CTR', 'CTS', 'DBC',
            'DCM', 'DGC', 'DGW', 'DIG', 'DPM', 'DSE', 'DXG', 'DXS', 'EIB', 'EVF', 'FPT', 'FRT', 'FTS', 'GAS', 'GEX', 'GMD', 'GVR',
            'HAG', 'HCM', 'HDB', 'HDC', 'HDG', 'HHV', 'HPG', 'HSG', 'HT1', 'IMP', 'KBC', 'KDC', 'KDH', 'KOS', 'LPB', 'MBB', 'MSB',
            'MSN', 'MWG', 'NAB', 'NKG', 'NLG', 'NT2', 'OCB', 'PAN', 'PC1', 'PDR', 'PHR', 'PLX', 'PNJ', 'POW', 'PPC', 'PTB', 'PVD',
            'PVT', 'REE', 'SAB', 'SBT', 'SCS', 'SHB', 'SIP', 'SJS', 'SSB', 'SSI', 'STB', 'SZC', 'TCB', 'TCH', 'TLG', 'TPB', 'VCB',
            'VCG', 'VCI', 'VGC', 'VHM', 'VIB', 'VIC', 'VIX', 'VJC', 'VND', 'VNM', 'VPB', 'VPI', 'VRE', 'VTP', 'VNINDEX'
        ]

    def execute_query(self, query: str, *params):
        """
        Execute a database query and return results.
        
        :param query: SQL query string
        :param params: Optional query parameters
        :return: Query results
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, params)
            result = cursor.fetchall()
            cursor.close()
            return result
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_stock_price(self, tickersymbol: str) -> pd.DataFrame:
        """
        Retrieve stock price data for a given ticker symbol.
        
        :param tickersymbol: Stock ticker symbol
        :return: DataFrame with stock price data
        """
        # Queries for different price types remain the same as in the original script
        open_price_query = f"""
        select o.datetime, o.tickersymbol, o.price as open
        from quote.open o
        where o.tickersymbol = '{tickersymbol}'
        order by o.datetime
        """
        open_price = pd.DataFrame(self.execute_query(open_price_query), columns=["datetime", "tickersymbol", "open"])

        close_price_query = f"""
        select c.datetime, c.tickersymbol, c.price as close
        from quote.close c
        where c.tickersymbol = '{tickersymbol}'
        order by c.datetime
        """
        close_price = pd.DataFrame(self.execute_query(close_price_query), columns=["datetime", "tickersymbol", "close"])

        high_price_query = f"""
        SELECT DISTINCT ON (DATE(h.datetime)) DATE(h.datetime) AS datetime, h.tickersymbol, h.price as high
        FROM quote.high h
        WHERE h.tickersymbol = '{tickersymbol}'
        ORDER BY DATE(h.datetime), h.datetime DESC;
        """
        high_price = pd.DataFrame(self.execute_query(high_price_query), columns=["datetime", "tickersymbol", "high"])

        low_price_query = f"""
        SELECT DISTINCT ON (DATE(l.datetime)) DATE(l.datetime) AS datetime, l.tickersymbol, l.price as low
        FROM quote.low l
        WHERE l.tickersymbol = '{tickersymbol}'
        ORDER BY DATE(l.datetime), l.datetime DESC;
        """
        low_price = pd.DataFrame(self.execute_query(low_price_query), columns=["datetime", "tickersymbol", "low"])

        quantity_query = f"""
        SELECT 
            DATE(c.datetime) AS datetime,
            c.tickersymbol AS tickersymbol,
            SUM(c.quantity) AS quantity
        FROM 
            quote.matchedvolume c
        WHERE 
            c.tickersymbol = '{tickersymbol}'
        GROUP BY 
            DATE(c.datetime), c.tickersymbol
        ORDER BY 
            DATE(c.datetime) DESC;
        """
        quantity = pd.DataFrame(self.execute_query(quantity_query), columns=["datetime", "tickersymbol", "quantity"])

        # Merge all price data
        stock_price = pd.merge(open_price, close_price, on=["datetime", "tickersymbol"], how='outer')
        stock_price = pd.merge(stock_price, high_price, on=["datetime", "tickersymbol"], how='outer')
        stock_price = pd.merge(stock_price, low_price, on=["datetime", "tickersymbol"], how='outer')
        stock_price = pd.merge(stock_price, quantity, on=["datetime", "tickersymbol"], how='outer')

        # Fill missing high/low values
        stock_price["low"] = stock_price.apply(lambda row: min(row["open"], row["close"]) if pd.isna(row["low"]) else row["low"], axis=1)
        stock_price["high"] = stock_price.apply(lambda row: max(row["open"], row["close"]) if pd.isna(row["high"]) else row["high"], axis=1)

        return stock_price

    def get_stock_price_data(self, tickers: List[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve and optionally save stock price data for given tickers.
        
        :param tickers: List of ticker symbols (defaults to VN 100 tickers)
        :param start_date: Optional start date for filtering
        :param end_date: Optional end date for filtering
        :return: DataFrame with stock price data
        """
        if tickers is None:
            tickers = self.vn_100_tickers

        stock_price_data = None
        for ticker in tickers:
            stock_price = self.get_stock_price(ticker)
            stock_price.to_csv(f"vn100/{ticker}.csv", index=False)
            print(f"Saved {ticker}.csv")
            
            # Concatenate dataframes if multiple tickers
            if stock_price_data is None:
                stock_price_data = stock_price
            else:
                stock_price_data = pd.concat([stock_price_data, stock_price], ignore_index=True)

        if start_date:
            stock_price_data = stock_price_data[stock_price_data["datetime"] >= start_date]
        if end_date:
            stock_price_data = stock_price_data[stock_price_data["datetime"] <= end_date]

        return stock_price_data

    def get_financial_statement_data(self) -> pd.DataFrame:
        """
        Retrieve and pivot financial statement data.
        
        :return: Pivoted financial data DataFrame
        """
        # Financial report fields query
        financial_report_fields_query = """
        select distinct info.code, item.name
        from financial.info info join financial.item item on info.code = item.code
        order by info.code
        """

        financial_report_fields = pd.DataFrame(
            self.execute_query(financial_report_fields_query), 
            columns=["code", "name"]
        )

        # Financial report data query
        financial_report_data_query = """
        select info.id, info.tickersymbol, info.year, info.quarter, info.value, info.code, item.name 
        from financial.info info join financial.item item on info.code = item.code
        order by info.year, info.quarter, info.tickersymbol, info.code
        """

        financial_report_data = pd.DataFrame(
            self.execute_query(financial_report_data_query), 
            columns=["id", "tickersymbol", "year", "quarter", "value", "code", "name"]
        )

        # Pivot the financial data
        pivoted_financial_data = financial_report_data.pivot_table(
            index=["tickersymbol", "year", "quarter"],
            columns="name",
            values="value",
            aggfunc="sum"
        ).reset_index()

        return pivoted_financial_data

    def close_connection(self):
        """
        Close the database connection.
        """
        if self.connection:
            self.connection.close()