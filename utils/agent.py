from google import genai
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

import datetime

today = datetime.datetime.now()


import yfinance as yf
from datetime import datetime, timedelta

def live_data_tool(stock_symbol: str):
    """
    Fetch historical stock data for the past 30 days using Yahoo Finance.

    Args:
        stock_symbol (str): Ticker symbol of the stock (e.g., 'AAPL', 'GOOG').

    Returns:
        pandas.DataFrame: Historical stock data for the last 30 days.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    
    ticker = yf.Ticker(stock_symbol)
    history = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    
    return history
from google.genai import types

config = types.GenerateContentConfig(
    tools=[live_data_tool]
)  # Pass the function itself











class StockInfo(BaseModel):
  stock_name: str
  stock_symbol: str








class StockAgent:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def get_stock_info(self, query):
        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=query,
            config={
        'response_mime_type': 'application/json',
        'response_schema': StockInfo,
        },
        )
        return response.parsed.stock_symbol
    
    def generate_final_answer(self, query, best_indicators, stock_symbol):
        content = f"""
        Answer the user query like a STOCK ANALYST.
        User query: {query}
        Top 10 best indicators: {best_indicators} 
        Historical/current data and current prices: live_data_tool - this tool takes stock_symbol: {stock_symbol} as argument.
        The date today is: {today}
        """
        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=content,
            config=config

        )
        return response.text
