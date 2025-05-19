import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

class StockPredictor:
    def __init__(self, numeric_threshold: float = 0.9):
        """
        numeric_threshold: fraction of parseable numeric values in an object column
                           above which we coerce it to numeric.
        """
        self.numeric_threshold = numeric_threshold
        self.cat_cols = []
        self.num_cols = []
        self.cat_imputer = None
        self.num_imputer = None
        self.encoder = None  # will be OrdinalEncoder
        self.scaler = None
        self.model = None

    def process_data(self, df: pd.DataFrame):
        # 1) TIME parsing & sorting with robust handling
        df = df.copy()
        # ensure string and zero-pad
        df['TimeOfScan_str'] = df['TimeOfScan'].astype(str).str.zfill(4)
        # try strict parse
        df['TimeOfScan_dt'] = pd.to_datetime(
            df['TimeOfScan_str'], format='%H%M', errors='coerce'
        )
        # fallback mixed parsing for any unparsable
        mask = df['TimeOfScan_dt'].isna()
        if mask.any():
            df.loc[mask, 'TimeOfScan_dt'] = pd.to_datetime(
                df.loc[mask, 'TimeOfScan_str'], format='mixed', errors='coerce'
            )
        # drop if still unparsable
        df = df.dropna(subset=['TimeOfScan_dt']).reset_index(drop=True)
        df['TimeOfScan_dt'] = df['TimeOfScan_dt'].dt.time
        df = df.sort_values(['column0', 'TimeOfScan_dt']).reset_index(drop=True)

        # 2) select predictors & create target
        good = [
            'DataName_Dominant_stock', 'DataName_ThisMove_stock', 'DataName_BullBearDir_stock',
            'DataName_OppSupRes_stock', 'DataName_Class_stock', 'DataName_Stock_Realtime_stock',
            'DataName_Correl_stock', 'DataName_PriorClass', 'DataName_IntTurn',
            'Timeframe_DailyMinute_stock', 'DataName_Dominant_sector', 'DataName_Class_sector',
            'DataName_BullBearDir_sector', 'BuySellSignal_GreenStronger', 'SupportResistanceLevel',
            'Sector', 'DataName_Class_Prior', 'DataGreenVal_SupportRange_sector',
            'DataRedVal_ResistanceRange_sector', 'Timeframe_DailyMinute_sector',
            'DataVal_PercChange_stock', 'DataVal_Price_Change', 'DataVal3_SupportPerc_stock',
            'DataVal4_ResistancePerc_stock', 'DataGreenVal_SupportRange_stock',
            'DataRedVal_ResistanceRange_stock', 'DataMagentaVal_MajorResistance',
            'DataYellowVal_MajorSupport', 'DataVal_PercDiff', 'ClosePrice_stock',
            'divergenceCount_stock', 'dailyPercentileHigh_stock', 'dailyPercentileLow_stock',
            'dailyPercentileCurrent_stock', 'dailyPercentileCount_stock', 'dailyPercentileSPY_stock',
            'EarlyMoveSignal',
            'divergenceCount_sector', 'dailyPercentileHigh_sector', 'dailyPercentileLow_sector',
            'dailyPercentileCurrent_sector', 'dailyPercentileCount_sector', 'dailyPercentileSPY_sector'
        ]
        df = df[good].copy()
        df['Future_closing_price_stock'] = df['ClosePrice_stock'].shift(-1)
        df = df.dropna(subset=['Future_closing_price_stock']).reset_index(drop=True)

        # target: +1 if next close higher, else -1
        df['Target'] = np.where(
            df['Future_closing_price_stock'] > df['ClosePrice_stock'],
            1, -1
        )

        # split X/y
        y = df['Target'].astype(int)
        X = df.drop(['Future_closing_price_stock', 'ClosePrice_stock', 'Target'], axis=1)

        # detect numeric-like object cols and coerce
        for col in X.columns:
            if X[col].dtype == object:
                conv = pd.to_numeric(X[col], errors='coerce')
                if conv.notna().mean() >= self.numeric_threshold:
                    X[col] = conv

        # identify types
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = [c for c in X.columns if c not in self.num_cols]

        # impute
        self.num_imputer = SimpleImputer(strategy='mean')
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='__MISSING__')

        X_num = pd.DataFrame(
            self.num_imputer.fit_transform(X[self.num_cols]),
            columns=self.num_cols, index=X.index
        )
        X_cat = pd.DataFrame(
            self.cat_imputer.fit_transform(X[self.cat_cols]),
            columns=self.cat_cols, index=X.index
        )

        # nominal encoding
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_cat_enc = pd.DataFrame(
            self.encoder.fit_transform(X_cat),
            columns=self.cat_cols, index=X.index
        )

        # scale numeric
        self.scaler = StandardScaler()
        X_num_sc = pd.DataFrame(
            self.scaler.fit_transform(X_num),
            columns=self.num_cols, index=X.index
        )

        # final matrix
        X_processed = pd.concat([X_cat_enc, X_num_sc], axis=1)
        return X_processed, y

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train logistic regression model on processed data.
        """
        self.model = LogisticRegression(solver='liblinear', max_iter=1000)
        self.model.fit(X, y)

    def get_best_combinations(self, X_processed: pd.DataFrame, top_n: int = 5):
        """
        Score each row by model probability for class=1 and return top_n rows decoded.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        # probabilities for positive class
        probs = self.model.predict_proba(X_processed)[:, 1]
        # select top indices
        best_idx = np.argsort(probs)[-top_n:][::-1]
        decoded = self.decode_rows(X_processed.iloc[best_idx])
        decoded['Buy_Probability'] = probs[best_idx]
        return decoded.reset_index(drop=True)

    def decode_rows(self, X_enc: pd.DataFrame):
        """
        Inverse transform encoded data back to original categories and unscaled numbers.
        """
        X_cat_enc = X_enc[self.cat_cols]
        X_num_enc = X_enc[self.num_cols]
        cat_dec = pd.DataFrame(
            self.encoder.inverse_transform(X_cat_enc),
            columns=self.cat_cols, index=X_enc.index
        )
        num_dec = pd.DataFrame(
            self.scaler.inverse_transform(X_num_enc),
            columns=self.num_cols, index=X_enc.index
        )
        return pd.concat([cat_dec, num_dec], axis=1)

# # ---------------------- USAGE EXAMPLE ----------------------
# if __name__ == "__main__":
#     # Generate a dummy DataFrame with required columns
#     n_rows = 100
#     np.random.seed(42)
#     df_dummy = pd.DataFrame({
#         'column0': np.random.choice(['A', 'B'], size=n_rows),
#         'TimeOfScan': np.random.randint(0, 2359, size=n_rows),
#         'DataName_Dominant_stock': np.random.choice(['X', 'Y', 'Z'], size=n_rows),
#         'DataName_ThisMove_stock': np.random.choice(['Up', 'Down'], size=n_rows),
#         'DataName_BullBearDir_stock': np.random.choice(['Bull', 'Bear'], size=n_rows),
#         'DataName_OppSupRes_stock': np.random.choice(['Sup', 'Res'], size=n_rows),
#         'DataName_Class_stock': np.random.choice(['C1', 'C2'], size=n_rows),
#         'DataName_Stock_Realtime_stock': np.random.choice(['R1', 'R2'], size=n_rows),
#         'DataName_Correl_stock': np.random.choice(['High', 'Low'], size=n_rows),
#         'DataName_PriorClass': np.random.choice(['P1', 'P2'], size=n_rows),
#         'DataName_IntTurn': np.random.choice(['T1', 'T2'], size=n_rows),
#         'Timeframe_DailyMinute_stock': np.random.choice(['D', 'M'], size=n_rows),
#         'DataName_Dominant_sector': np.random.choice(['S1', 'S2'], size=n_rows),
#         'DataName_Class_sector': np.random.choice(['SC1', 'SC2'], size=n_rows),
#         'DataName_BullBearDir_sector': np.random.choice(['Bull', 'Bear'], size=n_rows),
#         'BuySellSignal_GreenStronger': np.random.choice(['G', 'R'], size=n_rows),
#         'SupportResistanceLevel': np.random.rand(n_rows)*100,
#         'Sector': np.random.choice(['Tech', 'Finance'], size=n_rows),
#         'DataName_Class_Prior': np.random.choice(['CP1', 'CP2'], size=n_rows),
#         'DataGreenVal_SupportRange_sector': np.random.rand(n_rows)*50,
#         'DataRedVal_ResistanceRange_sector': np.random.rand(n_rows)*50,
#         'Timeframe_DailyMinute_sector': np.random.choice(['D', 'M'], size=n_rows),
#         # numeric columns
#         'DataVal_PercChange_stock': np.random.randn(n_rows),
#         'DataVal_Price_Change': np.random.randn(n_rows),
#         'DataVal3_SupportPerc_stock': np.random.randn(n_rows),
#         'DataVal4_ResistancePerc_stock': np.random.randn(n_rows),
#         'DataGreenVal_SupportRange_stock': np.random.randn(n_rows),
#         'DataRedVal_ResistanceRange_stock': np.random.randn(n_rows),
#         'DataMagentaVal_MajorResistance': np.random.randn(n_rows),
#         'DataYellowVal_MajorSupport': np.random.randn(n_rows),
#         'DataVal_PercDiff': np.random.randn(n_rows),
#         'ClosePrice_stock': np.random.rand(n_rows)*200,
#         'divergenceCount_stock': np.random.randint(0, 5, size=n_rows),
#         'dailyPercentileHigh_stock': np.random.rand(n_rows),
#         'dailyPercentileLow_stock': np.random.rand(n_rows),
#         'dailyPercentileCurrent_stock': np.random.rand(n_rows),
#         'dailyPercentileCount_stock': np.random.randint(0, 10, size=n_rows),
#         'dailyPercentileSPY_stock': np.random.rand(n_rows),
#         'EarlyMoveSignal': np.random.choice(['E1', 'E2'], size=n_rows),
#         'divergenceCount_sector': np.random.randint(0, 5, size=n_rows),
#         'dailyPercentileHigh_sector': np.random.rand(n_rows),
#         'dailyPercentileLow_sector': np.random.rand(n_rows),
#         'dailyPercentileCurrent_sector': np.random.rand(n_rows),
#         'dailyPercentileCount_sector': np.random.randint(0, 10, size=n_rows),
#         'dailyPercentileSPY_sector': np.random.rand(n_rows)
#     })

#     predictor = StockPredictor()
#     X_proc, y = predictor.process_data(df_dummy)
#     predictor.train(X_proc, y)
#     best = predictor.get_best_combinations(X_proc, top_n=10)
#     print(best)
