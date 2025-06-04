import pandas as pd
from sqlalchemy import create_engine
from dataclasses import dataclass

from scripts.date import get_next_date

@dataclass
class OptionKey:
    def __init__(self, symbol, expiration, strike):
        self.symbol = symbol
        self.expiration = expiration
        self.strike = strike

@dataclass
class Straddle:
    def __init__(self, call_id, put_id):
        self.call_id = call_id
        self.put_id = put_id

batch_size = 1000

date = get_next_date("")

engine = create_engine("mysql+pymysql://root:password@192.168.1.189/financial_data")

if date == "":
    raise Exception("No data")

while date != "":
    print("Starting date: " + date)

    chains = pd.read_sql_query("SELECT * FROM option_chain WHERE date = '" + date + "'", engine)

    print("Options found: ", len(chains))

    straddles = {}

    # Split into call and put options
    for call in chains[chains['option_type'] == 0].iterrows():
        for put in chains[chains['option_type'] == 1].iterrows():
            if call['symbol'] == put['symbol'] and call['strike'] == put['strike'] and call['expiration'] == put['expiration']:
                # straddles[OptionKey(call['symbol'], call['expiration'], call['strike'])] = Straddle(call['id'], put['id'])

    # Batch straddles and insert into database
    for i in range(0, len(straddles), batch_size):
        straddles_batch = list(straddles.values())[i:i+batch_size]
        straddles_df = pd.DataFrame(straddles_batch)
        straddles_df.to_sql('option_straddles', engine, if_exists='append', index=False)

    date = get_next_date(date)