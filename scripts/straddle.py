import pandas as pd
from sqlalchemy import create_engine
from dataclasses import dataclass

from scripts.date import get_next_date
from scripts.option_key import OptionKey


@dataclass
class Straddle:
    def __init__(self, call_option_id, put_option_id, current_date, expiration):
        self.call_option_id = call_option_id
        self.put_option_id = put_option_id
        self.date = current_date
        self.expiration = expiration

batch_size = 1000

date = get_next_date("")

engine = create_engine("mysql+pymysql://root:password@192.168.1.189/financial_data")

if date == "":
    raise Exception("No data")

while date != "":
    print("Date: " + date)

    chains = pd.read_sql_query("SELECT * FROM options WHERE date = '" + date + "'", engine)

    print("Options found: ", len(chains))

    straddles = {}

    calls = {}
    for call in chains[chains['option_type'] == 0].iterrows():
        calls[OptionKey(call[1]['symbol'], call[1]['expiration'], call[1]['strike'])] = call

    print("Calls found: ", len(calls))

    puts = {}
    for put in chains[chains['option_type'] == 1].iterrows():
        puts[OptionKey(put[1]['symbol'], put[1]['expiration'], put[1]['strike'])] = put

    print("Puts found: ", len(puts))

    for option_id in calls.keys():
        if option_id in puts:
            straddles[option_id] = Straddle(calls[option_id][1]['option_id'], puts[option_id][1]['option_id'], date, calls[option_id][1]['expiration'])

    data = []

    for straddle in straddles.values():
        data.append({
                'call_option_id': straddle.call_option_id,
                'put_option_id': straddle.put_option_id,
                'date': straddle.date,
                'expiration': straddle.expiration
            })

    print("Straddles matched: ", len(straddles))

    # Batch straddles and insert into database
    for i in range(0, len(data), batch_size):
        batch = list(data)[i:i + batch_size]
        batch_df = pd.DataFrame(batch)
        insert = batch_df.to_sql('option_straddle', engine, if_exists='append', index=False)

    date = get_next_date(date)