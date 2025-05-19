import os
import psycopg2
import pandas as pd
from psycopg2 import OperationalError
from dotenv import load_dotenv

load_dotenv()

pg_db = os.getenv('pg_db')
pg_user = os.getenv('pg_user')
pg_pass = os.getenv('pg_pass')
pg_host = os.getenv('pg_host')



class DataManager:
    # Rename mappings for stock and sector data
    STOCK_RENAME_MAP = {
        'column1': 'DataVal_PercChange',
        'column2': 'DataName_Dominant',
        'column3': 'DataName_ThisMove',
        'column4': 'DataVal_Price_Change',
        'column5': 'DataName_Symbol',
        'column6': 'DataName_BullBearDir',
        'column7': 'DataName_Support',
        'column8': 'DataName_OppSupRes',
        'column9': 'DataName_IndustryGroup',
        'column10': 'DataName_Sector',
        'column11': 'DataName_Class',
        'column12': 'DataVal3_SupportPerc',
        'column13': 'DataVal4_ResistancePerc',
        'column14': 'DataName_Stock_Realtime',
        'column15': 'DataName_Stock_Last',
        'column16': 'DataGreenVal_SupportRange',
        'column17': 'DataRedVal_ResistanceRange',
        'column18': 'DataMagentaVal_MajorResistance',
        'column19': 'DataYellowVal_MajorSupport',
        'column20': 'DataName_Age',
        'column21': 'DataName_Correl',
        'column22': 'DataName_PriorClass',
        'column23': 'DataName_IntTurn',
        'column24': 'DataVal_PercDiff',
        'column25': 'Timeframe_DailyMinute',
        'column26': 'ClosePrice',
        'column27': 'TimeOfScan'
    }

    SECTOR_RENAME_MAP = {
        'column1': 'DataName_Dominant',
        'column2': 'DataVal_PercChange',
        'column3': 'BuySellSignal_GreenStronger',
        'column4': 'EarlyMoveSignal',
        'column5': 'SubSectorSymbol',
        'column6': 'SubSectorName',
        'column7': 'SupportResistanceLevel',
        'column8': 'Sector',
        'column9': 'DataName_Class',
        'column10': 'DataVal3_SupportPerc',
        'column11': 'DataVal4_ResistancePerc',
        'column13': 'DataName_ThisMove',
        'column14': 'DataName_BullBearDir',
        'column15': 'DataName_Stock_Realtime',
        'column17': 'DataName_OppSupRes',
        'column18': 'DataName_Correl',
        'column19': 'DataName_Class_Prior',
        'column20': 'OpenCloseCombo',
        'column21': 'DataGreenVal_SupportRange',
        'column22': 'DataRedVal_ResistanceRange',
        'column25': 'Timeframe_DailyMinute',
        'column26': 'ClosePrice',
        'column27': 'TimeOfScan'
    }

    def __init__(self, stock_name: str):
        self.stock_name = stock_name

    def _create_connection(self):
        try:
            return psycopg2.connect(
                dbname=pg_db,
                user=pg_user,
                password=pg_pass,
                host=pg_host,
                port=5432
            )
        except OperationalError as e:
            raise ConnectionError(f"Database connection failed: {e}")

    def _fetch_data(self, filter_col: str, filter_val: str) -> pd.DataFrame:
        conn = self._create_connection()
        try:
            query = 'SELECT * FROM "MarketData" WHERE {col} = %s;'.format(col=filter_col)
            df = pd.read_sql(query, conn, params=(filter_val,))
            if df.empty:
                raise ValueError(f"No data found for {filter_col} = {filter_val}")
            return df
        finally:
            conn.close()

    def _rename_columns(self, df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
        return df.rename(columns=rename_map)

    def _get_stock_df(self) -> pd.DataFrame:
        df_stock = self._fetch_data('column5', self.stock_name)
        df_stock = self._rename_columns(df_stock, self.STOCK_RENAME_MAP)
        return df_stock

    def _get_sector_df(self, subsector: str) -> pd.DataFrame:
        df_sector = self._fetch_data('column10', subsector)
        df_sector = self._rename_columns(df_sector, self.SECTOR_RENAME_MAP)
        # drop duplicates based on id and timestamp
        df_sector = df_sector.drop_duplicates(subset=['column0', 'TimeOfScan'])
        return df_sector

    def get_merged_data(self) -> pd.DataFrame:
        # Fetch and prepare stock data
        df_stock = self._get_stock_df()
        # Identify subsector from the stock data
        subsector = df_stock['DataName_Sector'].iloc[0]
        # Fetch and prepare sector data
        df_sector = self._get_sector_df(subsector)
        # Merge with suffixes for overlapping columns
        merged_df = pd.merge(
            df_stock,
            df_sector,
            how='left',
            on=['column0', 'TimeOfScan'],
            suffixes=('_stock', '_sector')
        )
        merged_df.column0 = pd.to_datetime(merged_df.column0)
        # Sort by timestamp or any other columns
        return merged_df


manager = DataManager('GOOGL')
df = manager.get_merged_data()
print(df.head())
