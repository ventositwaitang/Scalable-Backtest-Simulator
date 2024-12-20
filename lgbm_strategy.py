from src.strategy import Strategy
import pandas as pd
from data_processor import DataLoader1
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from datetime import datetime
import numpy as np

class MyLGBMStrategy(Strategy):
    def __init__(self, broker, dataloader, neutralize=True, label='log_mid', model='lgbm'):
        super().__init__(broker, neutralize, model)
        self.xy = None
        self.dataloader = dataloader
        self.neutralize = neutralize
        self.factors = None
        self.data = None
        self.market = None
        self.label = label
        self.all_labels = ['log_close', 'log_open', 'log_mid', 'log_vwap', 'log_twap_mid']
        self.warm_date = np.datetime64('2017-09-30')
        self.warm_up = False
        self.train_x = None
        self.train_y = None
        self.model = lgb
        self.bst = None
        self.model_params = {
                            'objective': 'regression',  # Regression task
                            'metric': 'mse',
                            'num_leaves': 31,
                            'learning_rate': 0.05,
                            'feature_fraction': 0.9,
                            'bagging_fraction': 0.8,
                            'bagging_freq': 5,
                            'verbose': 0
                        }

    def init(self):
        self.data = self.dataloader.data
        if self.neutralize:
            self.factors = self.dataloader.neutralize()
        else:
            self.factors = self.dataloader.factors
        self.xy = self.dataloader.make_xy()
        self.train_y = self.xy.loc[self.xy.index < self.warm_date, 'y']
        self.train_x = self.xy.loc[self.xy.index < self.warm_date].drop(['ticker', 'y'], axis=1)

    def _warm_up(self):

        warmup_x = self.train_x.to_numpy()
        warmup_y = self.train_y.to_numpy()
        train_data = lgb.Dataset(warmup_x, label=warmup_y)
        self.bst = self.model.train(self.model_params, train_data, 50)

    # @nb.jit(parallel=True)
    def next(self, tick):
        if tick < self.warm_date:
            return
        if not self.warm_up:
            self._warm_up()
        else:
            # train_data = self.train_data.filter((pl.col('Time') < tick))
            self.train_x = self.xy.loc[self.xy.index < tick].drop(['ticker', 'y'], axis=1).to_numpy()

            self.train_y = self.xy.loc[self.xy.index < tick].y.to_numpy().reshape(-1)

            train_data = lgb.Dataset(self.train_x, label=self.train_y)
            self.bst = self.model.train(self.model_params, train_data, 50)

        tick_data = self.data.loc[tick]
        if not len(tick_data):
            return

        tick_x = self.xy.loc[tick].drop(['ticker', 'y'], axis=1).to_numpy()
        signal = self.bst.predict(tick_x, num_iteration=self.bst.best_iteration)

        # split into 5 groups based on signal
        signal = pd.Series(signal, index=self.xy.loc[tick].ticker)
        signal = pd.qcut(signal, 10, labels=False, duplicates='drop')
        # buy the top 10% stocks
        # buy_stocks = tick_data.filter
        buy_stocks = signal[signal == 9].index.tolist()
        # sell the bottom 10% stocks
        sell_stocks = signal[signal == 0].index.tolist()

        for stock in sell_stocks:
            self.sell(stock)

        # determine the position level
        if self._broker.cash / self._broker.market_value < 0.2:
            return

        ava_cash = self._broker.cash - 0.2 * self._broker.market_value

        for stock in buy_stocks:
            amount = ava_cash / len(buy_stocks)
            p = tick_data.loc[tick_data.ticker == stock, 'last'].values[0]
            self.buy(stock, int(amount/p))


