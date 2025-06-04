import pandas as pd
from sqlalchemy import create_engine

def get_next_date(current):
    engine = create_engine("mysql+pymysql://root:password@192.168.1.189/financial_data")
    if current == "":
        df = pd.read_sql_query("SELECT MIN(date) as 'date' FROM option_chain ORDER BY date DESC LIMIT 1", engine)
        if df.empty:
            return ""
        return df['date'][0].strftime('%Y-%m-%d')
    else:
        df = pd.read_sql_query("SELECT MIN(date) as 'date' FROM option_chain WHERE date > '" + current + "'", engine)
        if df.empty:
            return ""
        return df['date'][0].strftime('%Y-%m-%d')