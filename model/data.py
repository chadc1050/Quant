import pandas as pd
from sqlalchemy import create_engine


def get_data():
    engine = create_engine("mysql+pymysql://root:password@192.168.1.189/financial_data")

    vix_df = pd.read_sql_query("SELECT * FROM vix", engine)
    vix_df.dropna(subset=["index_value"], inplace=True)
    vix_df['observation_date'] = pd.to_datetime(vix_df['observation_date'])
    vix_df.sort_values('observation_date', inplace=True)

    treasury_df = pd.read_sql_query("SELECT * FROM treasury", engine)
    treasury_df['observation_date'] = pd.to_datetime(treasury_df['observation_date'])
    treasury_df.sort_values('observation_date', inplace=True)

    unemploy_df = pd.read_sql_query("SELECT * FROM unemployment", engine)
    unemploy_df['observation_date'] = pd.to_datetime(unemploy_df['observation_date'])
    unemploy_df.sort_values('observation_date', inplace=True)

    merged = pd.merge_asof(vix_df, treasury_df, on='observation_date', direction='backward')
    merged = pd.merge_asof(merged, unemploy_df, on='observation_date', direction='backward')

    return merged