import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
        self.encoder = None
        self.scaler = None
        self.model = None
        self.ohe_feature_names = None

    def process_data(self, df: pd.DataFrame):
        # 1) TIME parsing & sorting
        df = df.copy()
        df['TimeOfScan'] = df['TimeOfScan'].astype(str).str.zfill(4)
        df['TimeOfScan_dt'] = pd.to_datetime(df['TimeOfScan'], format='%H%M').dt.time
        df = df.sort_values(['column0','TimeOfScan_dt']).reset_index(drop=True)

        # 2) filter predictors & create target
        good = [
            # categorical predictors
            'DataName_Dominant_stock', 'DataName_ThisMove_stock', 'DataName_BullBearDir_stock',
            'DataName_OppSupRes_stock', 'DataName_Class_stock', 'DataName_Stock_Realtime_stock',
            'DataName_Correl_stock', 'DataName_PriorClass', 'DataName_IntTurn',
            'Timeframe_DailyMinute_stock', 'DataName_Dominant_sector', 'DataName_Class_sector',
            'DataName_BullBearDir_sector', 'BuySellSignal_GreenStronger', 'SupportResistanceLevel',
            'Sector', 'DataName_Class_Prior', 'DataGreenVal_SupportRange_sector',
            'DataRedVal_ResistanceRange_sector', 'Timeframe_DailyMinute_sector',
            # numeric predictors
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

        # create target labels
        df['Target'] = np.where(
            pd.to_numeric(df['Future_closing_price_stock'], errors='coerce').fillna(0) >
            pd.to_numeric(df['ClosePrice_stock'], errors='coerce').fillna(0),
            1, -1
        )

        # 3) split X/y
        y = df['Target'].astype(int)
        X = df.drop(['Future_closing_price_stock', 'ClosePrice_stock', 'Target'], axis=1)

        # 4) detect numeric-like object columns and coerce
        for col in X.columns:
            if X[col].dtype == object:
                converted = pd.to_numeric(X[col], errors='coerce')
                frac_numeric = converted.notna().mean()
                if frac_numeric >= self.numeric_threshold:
                    X[col] = converted

        # 5) identify final column types
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = [c for c in X.columns if c not in self.num_cols]

        # 6) impute missing values
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

        # 7) encoding + scaling
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat_enc = pd.DataFrame(
            self.encoder.fit_transform(X_cat),
            columns=self.encoder.get_feature_names_out(self.cat_cols),
            index=X.index
        )
        self.ohe_feature_names = X_cat_enc.columns.tolist()

        self.scaler = StandardScaler()
        X_num_sc = pd.DataFrame(
            self.scaler.fit_transform(X_num),
            columns=self.num_cols, index=X.index
        )

        # final feature matrix
        X_processed = pd.concat([X_cat_enc, X_num_sc], axis=1)
        return X_processed, y

    def predict_features(self, X: pd.DataFrame, y: pd.Series, top_n: int = 10):
        """
        Train a logistic model and return top_n positive and negative contributing features.
        """
        self.model = LogisticRegression(solver='liblinear', max_iter=1000)
        self.model.fit(X, y)

        coefs = pd.Series(self.model.coef_[0], index=X.columns)
        top_pos = coefs.nlargest(top_n)
        top_neg = coefs.nsmallest(top_n)
        return top_pos, top_neg

    def decode_row(self, X_enc_row: pd.DataFrame):
        """
        Inverse-transform one-hot + scaled numeric back to original values.
        """
        # split blocks
        cat_block = X_enc_row[self.ohe_feature_names]
        num_block = X_enc_row.drop(columns=self.ohe_feature_names)

        cat_decoded = pd.DataFrame(
            self.encoder.inverse_transform(cat_block),
            columns=self.cat_cols, index=cat_block.index
        )
        num_decoded = pd.DataFrame(
            self.scaler.inverse_transform(num_block),
            columns=self.num_cols, index=num_block.index
        )

        return pd.concat([cat_decoded, num_decoded], axis=1)


