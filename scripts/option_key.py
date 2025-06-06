from dataclasses import dataclass

@dataclass
class OptionKey:
    def __init__(self, symbol, expiration, strike):
        self.symbol = symbol
        self.expiration = expiration
        self.strike = strike

    def __hash__(self):
        return hash((self.symbol, self.expiration, self.strike))
