from  utils.data_manager import DataManager
from utils.stock_predictor import StockPredictor



manager = DataManager('GOOGL')
df = manager.get_merged_data()



# 2) initialize & process
sp = StockPredictor(numeric_threshold=0.9)
X, y = sp.process_data(df)


# 3) fit & extract top contributors
top_up, top_down = sp.predict_features(X, y, top_n=10)

print("Top 10 features ↑ target:")
print(top_up.to_string())

print("\nTop 10 features ↓ target:")
print(top_down.to_string())



