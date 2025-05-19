import streamlit as st
from utils.data_manager import DataManager
# from utils.stock_predictor import StockPredictor
# from utils.stock_predictor_1 import BuySignalPredictor
from utils.stock_predictor_1 import StockPredictor
from utils.agent import StockAgent
import time
import pandas as pd

st.set_page_config(page_title="Stock Predictor", page_icon="📈")
st.title("Stock Market Analysis")

# Initialize the agent
agent = StockAgent()

# Create a text input for the user's query
query = st.text_input("Enter your stock-related question:", placeholder="E.g., What will happen to Apple stock tomorrow?")

if st.button("Analyze"):
    if query:
        try:
            with st.spinner("Agent extracting the stock symbol..."):
                # Get stock symbol from query
                stock_symbol = agent.get_stock_info(query)
                if not stock_symbol:
                    st.error("Could not detect a stock symbol from your query. Please try again with a clearer question.")
                    st.stop()
                st.info(f"Detected stock symbol: {stock_symbol}")
                time.sleep(1)  # Small delay for better user experience
            
            try:
                with st.spinner("Agent querying the database..."):
                    # Get stock data
                    manager = DataManager(stock_symbol)
                    df = manager.get_merged_data()
                    if df.empty:
                        st.error(f"No data found for stock symbol: {stock_symbol}")
                        st.stop()
                    st.success("Data successfully retrieved!")
                    time.sleep(1)
                
                with st.spinner("Agent processing indicators..."):
                    # Process data
                    # sp = StockPredictor(numeric_threshold=0.9)
                    # sp = BuySignalPredictor(numeric_threshold=0.9)

                    predictor = StockPredictor()
                    X_proc, y = predictor.process_data(df)
                    st.success("Data processed successfully!")
                    predictor.train(X_proc, y)
                    


                    # X, y = sp.process_data(df)
                    # st.success("Data processed successfully!")
                    # time.sleep(1)
                
                with st.spinner("Agent identifying top indicators..."):
                    # Extract top contributors
                    # top_up, top_down = sp.predict_features(X, y, top_n=10)
                    best = predictor.get_best_combinations(X_proc, top_n=10)
                    
                    # Display top indicators in expanders
                    with st.expander("Top 10 Best buy indicators"):
                        st.dataframe(best)
                    
                    # with st.expander("Top 10 indicators suggesting price decrease"):
                    #     st.dataframe(top_down)
                    
                    time.sleep(1)
                
                with st.spinner("Agent generating final analysis..."):
                    # Generate the final answer
                    final_answer = agent.generate_final_answer(query, top_up, top_down)
                    if not final_answer:
                        st.warning("The analysis completed, but no detailed explanation could be generated.")
                    time.sleep(1)
                
                # Display the final answer
                st.markdown("## Analysis Results")
                st.markdown(final_answer)
            
            except ConnectionError as e:
                st.error(f"Database connection error: {str(e)}")
            except ValueError as e:
                st.error(f"Data processing error: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
        
        except Exception as e:
            st.error(f"Error extracting stock symbol: {str(e)}")
    else:
        st.error("Please enter a query to analyze.")



