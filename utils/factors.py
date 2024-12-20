import pandas as pd
import numpy as np
from src.factors import *

class FactorMaker:
    def __init__(self, data):
        self.data = data
        self.data.set_index('date', inplace=True)

    def make_factor(self):
        factors = []
        for ticker, data in self.data.groupby('ticker'):
            factor = self._make_factor(data.sort_index())
            factor['ticker'] = ticker
            factors.append(factor)
        factors = pd.concat(factors).reset_index()
        return factors

    def _make_factor(self, data):
        factor = pd.DataFrame(index=data.index)
        factor['vwap'] = VWAP(data)
        factor['mom'] = MOM(data['last'])
        factor['sma'] = SMA(data['last'])
        factor['rsi'] = RSI(data['last'])
        factor['obv'] = OBV(data['last'])
        factor = MACD(factor, data['last'])
        factor = Reg_factors(factor, data['last'])
        return factor
