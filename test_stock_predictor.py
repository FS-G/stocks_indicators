import pandas as pd
import numpy as np
from utils.stock_predictor_1 import BuySignalPredictor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, time
import random

def generate_sample_data(n_samples=1000):
    """
    Generate sample stock data that mimics the structure of real stock data.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate base data
    data = {
        'column0': [f'STOCK_{i%10}' for i in range(n_samples)],  # 10 different stocks
        'TimeOfScan': [f"{random.randint(0,23):02d}{random.randint(0,59):02d}" for _ in range(n_samples)],
        'ClosePrice_stock': np.random.normal(100, 10, n_samples).cumsum() + 1000,  # Random walk
    }
    
    # Generate categorical features
    categorical_features = {
        'DataName_Dominant_stock': ['BULL', 'BEAR', 'NEUTRAL'],
        'DataName_ThisMove_stock': ['UP', 'DOWN', 'SIDEWAYS'],
        'DataName_BullBearDir_stock': ['BULLISH', 'BEARISH', 'NEUTRAL'],
        'DataName_OppSupRes_stock': ['SUPPORT', 'RESISTANCE', 'NONE'],
        'DataName_Class_stock': ['STRONG', 'WEAK', 'MODERATE'],
        'Sector': ['TECH', 'FINANCE', 'HEALTHCARE', 'ENERGY', 'CONSUMER'],
        'BuySellSignal_GreenStronger': ['YES', 'NO'],
        'SupportResistanceLevel': ['STRONG', 'WEAK', 'NONE']
    }
    
    for feature, categories in categorical_features.items():
        data[feature] = [random.choice(categories) for _ in range(n_samples)]
    
    # Generate numeric features with some correlation to price movement
    price_changes = np.diff(data['ClosePrice_stock'], prepend=data['ClosePrice_stock'][0])
    
    numeric_features = {
        'DataVal_PercChange_stock': price_changes / data['ClosePrice_stock'] * 100,
        'DataVal_Price_Change': price_changes,
        'DataVal3_SupportPerc_stock': np.random.normal(0, 1, n_samples),
        'DataVal4_ResistancePerc_stock': np.random.normal(0, 1, n_samples),
        'DataGreenVal_SupportRange_stock': np.random.normal(0, 1, n_samples),
        'DataRedVal_ResistanceRange_stock': np.random.normal(0, 1, n_samples),
        'divergenceCount_stock': np.random.randint(0, 5, n_samples),
        'dailyPercentileHigh_stock': np.random.uniform(0, 100, n_samples),
        'dailyPercentileLow_stock': np.random.uniform(0, 100, n_samples),
        'dailyPercentileCurrent_stock': np.random.uniform(0, 100, n_samples),
        'EarlyMoveSignal': np.random.choice([-1, 0, 1], n_samples)
    }
    
    # Add some correlation to price movement
    for feature in ['DataVal3_SupportPerc_stock', 'DataVal4_ResistancePerc_stock']:
        numeric_features[feature] = numeric_features[feature] + 0.3 * price_changes
    
    data.update(numeric_features)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values to make it more realistic
    for col in df.columns:
        if col not in ['column0', 'TimeOfScan', 'ClosePrice_stock']:
            mask = np.random.random(n_samples) < 0.05  # 5% missing values
            df.loc[mask, col] = np.nan
    
    return df

def test_predictor():
    """
    Test the BuySignalPredictor with sample data and visualize results.
    """
    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data(n_samples=2000)
    print(f"Generated {len(df)} samples")
    
    # Initialize and train predictor
    print("\nInitializing predictor...")
    predictor = BuySignalPredictor()
    
    print("Processing data...")
    X_processed, y = predictor.process_data(df)
    print(f"Processed data shape: {X_processed.shape}")
    print(f"Target distribution: {pd.Series(y).value_counts(normalize=True).to_dict()}")
    
    print("\nFinding best feature combination...")
    metrics = predictor.find_best_feature_combination(
        X_processed, y,
        min_features=3,
        max_features=8,
        n_trials=50  # Reduced for faster testing
    )
    
    # Print results
    print("\nResults:")
    print("-" * 50)
    print(f"Best feature combination: {metrics['best_features']}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    
    # Get feature importance
    importance_df = predictor.get_feature_importance()
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    
    # Plot feature importance
    plt.subplot(1, 2, 1)
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title('Feature Importance in Best Combination')
    plt.xlabel('Importance Score')
    
    # Plot precision-recall tradeoff
    plt.subplot(1, 2, 2)
    plt.scatter(metrics['recall'], metrics['precision'], s=100, c='blue')
    plt.title('Precision-Recall Tradeoff')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('buy_signal_analysis.png')
    print("\nResults visualization saved as 'buy_signal_analysis.png'")
    
    # Test predictions
    print("\nTesting predictions on a sample...")
    sample_data = df.sample(5)  # Get 5 random samples
    X_sample, _ = predictor.process_data(sample_data)
    predictions, probabilities = predictor.predict_buy_signal(X_sample)
    
    print("\nSample predictions:")
    print("-" * 50)
    results_df = pd.DataFrame({
        'Stock': sample_data['column0'],
        'Time': sample_data['TimeOfScan'],
        'Current Price': sample_data['ClosePrice_stock'],
        'Buy Signal': predictions,
        'Buy Probability': probabilities
    })
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    test_predictor() 