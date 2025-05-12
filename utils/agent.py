from google import genai
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

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
    
    def generate_final_answer(self, query, indicators_up, indicators_down):
        content = f"""

        Analyze the following indicators and tell in detiails the top ten indicators that will affect the stock price in up and down direction.
        JUST A BRIEF COMPREHENSIVE SUMMARY - TOP TEN INDICATORS THAT WILL AFFECT THE STOCK PRICE IN UP AND DOWN DIRECTION.
        Also make a tabular summary.
        Up indicators: {indicators_up}
        Down indicators: {indicators_down}

        NOTE: BE CONCISE - DONT SAY ANY OTHER THINGS. AND YOU SHOULD WRITE LIKE STOCKS ANALYST.


        """
        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=content,

        )
        return response.text
