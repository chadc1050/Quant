import datetime
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from scripts.date import get_next_date, get_start_date
from scripts.option_key import OptionKey

@dataclass
class StraddleDerived:
    def __init__(self, curr_date, straddle_id):
        self.option_straddle_id: int = straddle_id
        self.date = curr_date
        self.std5: Optional[float] = None
        self.std10: Optional[float] = None
        self.std15: Optional[float] = None
        self.std20: Optional[float] = None
        self.std25: Optional[float] = None
        self.std30: Optional[float] = None
        self.ret5: Optional[float] = None
        self.ret10: Optional[float] = None
        self.ret15: Optional[float] = None
        self.ret20: Optional[float] = None
        self.ret25: Optional[float] = None
        self.ret30: Optional[float] = None
        self.sma2: Optional[float] = None
        self.sma4: Optional[float] = None
        self.sma8: Optional[float] = None
        self.sma16: Optional[float] = None
        self.sma32: Optional[float] = None
        self.ema2: Optional[float] = None
        self.ema4: Optional[float] = None
        self.ema8: Optional[float] = None
        self.ema16: Optional[float] = None
        self.ema32: Optional[float] = None

def get_half_life(t: int) -> float:
    return np.log(0.5) / np.log(1 - 1 / t)

def get_return(values: list[float]) -> float:
    # -99999 will indicate arbitrage assumption violation
    if values[-1] == 0:
        return -99999.99

    return (values[0] - values[-1]) / values[-1]

def get_ema(values: list[float], t: int) -> float:
    return pd.DataFrame(reversed(values), columns=['price']).ewm(halflife=get_half_life(t), adjust=False).mean().iloc[-1, 0]

batch_size = 1000

date = get_start_date()

write_date = "2022-08-17"

engine = create_engine("mysql+pymysql://root:password@192.168.1.189/financial_data")

if date == "":
    raise Exception("No data")

date_window = []

while date != "":

    print("Date: " + date)

    date_window.append(date)

    # If we are not writing yet continue
    if write_date != "" and datetime.datetime.strptime(date, "%Y-%m-%d") < datetime.datetime.strptime(write_date, "%Y-%m-%d"):
        date = get_next_date(date)
        continue

    # Query 32-day lookback for straddles
    dates_in = ",".join(["'" + prevDate + "'" for prevDate in date_window[-32:]])

    df_straddles = pd.read_sql_query("SELECT * FROM straddle_view WHERE date in (" + dates_in + ")", engine)

    print("Straddles found: ", len(df_straddles))

    breakdown = {}

    for straddle in df_straddles.iterrows():
        key = OptionKey(straddle[1]['symbol'], straddle[1]['expiration'], straddle[1]['strike'])
        straddle_date = straddle[1]['date']
        if straddle_date not in breakdown:
            breakdown[straddle_date] = {}
        breakdown[straddle_date][key] = straddle

    insert = []
    for item in breakdown[datetime.datetime.strptime(date, "%Y-%m-%d").date()].keys():
        prices = []
        derived = StraddleDerived(date, breakdown[datetime.datetime.strptime(date, "%Y-%m-%d").date()][item][1]['option_straddle_id'])
        for lag in range(1, 33):
            if lag > len(date_window):
                break

            # If current date straddle is not defined at lag, then continue to next as it is no longer continuous
            if item not in breakdown[datetime.datetime.strptime(date_window[-lag], "%Y-%m-%d").date()]:
                break

            lag_rec = breakdown[datetime.datetime.strptime(date_window[-lag], "%Y-%m-%d").date()][item]
            price = (lag_rec[1]['call_bid'] + lag_rec[1]['call_ask']) / 2 + (lag_rec[1]['put_bid'] + lag_rec[1]['put_ask']) / 2

            prices.insert(0, price)

            # If midpoint price is zero, then skip, we may want to break out in the future as this would violate arbitrage assumptions
            if price == 0:
                continue

            # Calculate derived values
            if lag == 2:
                derived.sma2 = sum(prices) / 2
                derived.ema2 = get_ema(prices, 2)
            elif lag == 4:
                derived.sma4 = sum(prices) / 4
                derived.ema4 = get_ema(prices, 4)
            elif lag == 5:
                derived.ret5 = get_return(prices)
                derived.std5 = float(np.std(prices))
            elif lag == 8:
                derived.sma8 = sum(prices) / 8
                derived.ema8 = get_ema(prices, 8)
            elif lag == 10:
                derived.ret10 = get_return(prices)
                derived.std10 = float(np.std(prices))
            elif lag == 15:
                derived.ret15 = get_return(prices)
                derived.std15 = float(np.std(prices))
            if lag == 16:
                derived.sma16 = sum(prices) / 16
                derived.ema16 = get_ema(prices, 16)
            elif lag == 20:
                derived.ret20 = get_return(prices)
                derived.std20 = float(np.std(prices))
            elif lag == 25:
                derived.ret25 = get_return(prices)
                derived.std25 = float(np.std(prices))
            elif lag == 30:
                derived.ret30 = get_return(prices)
                derived.std30 = float(np.std(prices))
            elif lag == 32:
                derived.sma32 = sum(prices) / 32
                derived.ema32 = get_ema(prices, 32)
        insert.append(derived)
    print("Derived values: ", len(insert))

    data = []
    for insertion in insert:
        data.append({
            'straddle_id': insertion.option_straddle_id,
            'date': insertion.date,
            'std5': insertion.std5,
            'std10': insertion.std10,
            'std15': insertion.std15,
            'std20': insertion.std20,
            'std25': insertion.std25,
            'std30': insertion.std30,
            'ret5': insertion.ret5,
            'ret10': insertion.ret10,
            'ret15': insertion.ret15,
            'ret20': insertion.ret20,
            'ret25': insertion.ret25,
            'ret30': insertion.ret30,
            'sma2': insertion.sma2,
            'sma4': insertion.sma4,
            'sma8': insertion.sma8,
            'sma16': insertion.sma16,
            'sma32': insertion.sma32,
            'ema2': insertion.ema2,
            'ema4': insertion.ema4,
            'ema8': insertion.ema8,
            'ema16': insertion.ema16,
            'ema32': insertion.ema32
        })

    data_df = pd.DataFrame(data)
    data_df.to_sql('option_straddle_derived', engine, if_exists='append', index=False, chunksize=batch_size)

    date = get_next_date(date)