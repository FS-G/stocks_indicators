from datetime import datetime, timedelta
import yfinance as yf

def live_data_tool(stock_symbol: str):
    """
    Fetch historical stock data for the past 30 days using Yahoo Finance.

    Args:
        stock_symbol (str): Ticker symbol of the stock (e.g., 'AAPL', 'GOOG').

    Returns:
        pandas.DataFrame: Historical stock data for the last 30 days. Cas you used for the current price.
    """
    print(f"Fetching historical stock data for {stock_symbol}...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5)
    
    ticker = yf.Ticker(stock_symbol)
    history = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    history = history.to_json()
    print("history", history)
    
    return history



print(live_data_tool("GOOG"))