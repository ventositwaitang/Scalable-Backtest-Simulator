import pandas as pd
from .strategy import Strategy
from .broker import Broker
from tqdm import tqdm
import numpy as np
import os

class Backtest:
    """
    Backtest class is used for reading historical market data,
    executing strategies, simulating trades, and estimating returns.
    Call Backtest.run during initialization to initiate the backtesting process.
    """

    def __init__(self,
                 data_loader,
                 strategy: Strategy,
                 broker: Broker,
                 name: str):
        """
        Construct backtesting object. Required parameters include: historical data,
        strategy object, initial capital, commission rate, etc.
        The initialization process includes checking input types, filling data null values, etc.
        Parameters:
        :param data:           pd.DataFrame        Historical data in pandas DataFrame format
        :param broker:         type(Broker)        Broker type responsible for executing buy and sell operations
                                                   as well as maintaining account status.
        :param strategy:       type(Strategy)      Strategy Type
        :param cash:           float               Initial funding amount
        :param commission:     float               Transaction fee rate for each transaction. For example, if the fee is
                                                   .2%, then it should be entered as 0.002 here.
        """
        data = data_loader.data.copy(False)

        # Sort the market data by time if it is not already sorted.
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()
        # Initialize exchange and strategy objects using data.
        self._data = data.reset_index()[['date', 'ticker', 'last']]
        self.ticks = self._data['date'].unique()

        self._broker = broker
        self._strategy = strategy
        self._results = None
        self.name = name

    def run(self):
        """
        Run backtesting, iterate through historical data, execute simulated trades, and return backtesting results.
        Run the backtest. Returns `pd.Series` with results and statistics.
        Keyword arguments are interpreted as strategy parameters.
        """
        strategy = self._strategy
        broker = self._broker
        # Strategy Initialization
        strategy.init()
        # Set the start and end positions for backtesting
        # Backtesting main loop, update market status, and execute strategy

        for tick in tqdm(self.ticks):
            # tick_data = self._data.loc[self._data['date'] == tick]
            broker.next(tick)
            strategy.next(tick)
            broker.write_ratio(tick)
            # self.write_ratio()
        # After completing the strategy execution, calculate the results and return them.
        res = broker.get_result()
        n = len(os.listdir('./results'))
        broker.save_result(path='./results/backtest_{}_run_{}'.format(self.name, n+1))
        broker.plot_ratio(path='./results/backtest_{}_run_{}'.format(self.name, n+1))
        return res

