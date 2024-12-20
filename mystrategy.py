from src.strategy import Strategy
import pandas as pd
from data_processor import DataLoader
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

class MyStrategy(Strategy):
    def __init__(self, broker, risk_manage=True, model='random_forest'):
        super().__init__(broker, risk_manage, model)
        self.y = None
        self.x = None
        self.factors = None
        self._data = None
        self.pool = None
        self.last_tick = None
        self.grids = {}
        self.max_drawdown = 0.15
        self.take_profit_percent = 0.2
        self.stop_loss_percent = 0.05
        self.risk_manage = risk_manage
        if model == 'ridge':
            self.model = Ridge(alpha=0.5)
        elif model == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=20, max_depth=3)
        else:
            self.model = LinearRegression()

    def init(self):
        self.factors = self._broker.data_loader.factors
        # self.factors.index = self.factors['date']
        self._data = self._broker.data_loader.data

        self._data = pd.concat([self._data.set_index(['ticker', 'date']), self.factors], axis=1).reset_index().set_index('date')
        self.pool = self._data['ticker'].unique()

    def get_grids(self, tick, tick_data):
        if len(self.grids)==0 or datetime.fromtimestamp(tick.astype(datetime)/10**9).isoweekday() == 1:
            for p in self.pool:
                tmp = tick_data.loc[tick_data.ticker == p]
                upper = tmp.upper.values[0]
                lower = tmp.lower.values[0]
                grid_size = (upper - lower) / 10
                self.grids[p] = pd.Series([lower + i * grid_size for i in range(10 + 1)])
        return self.grids

    def gen_signal(self, crt_data, last_data):
        crt_data['signal'] = 0
        # find out the stock belong to which grid
        for p in crt_data.ticker:
            crt_price = crt_data.loc[crt_data.ticker == p]['last'].values[0]
            last_price = last_data.loc[last_data.ticker == p]['last'].values[0]
            last_idx = self.grids[p].searchsorted(last_price)
            crt_idx = self.grids[p].searchsorted(crt_price)
            crt_data.loc[crt_data.ticker == p, 'signal'] = last_idx - crt_idx
        return crt_data[['signal', 'ticker']].set_index('ticker')

    def next(self, tick):
        tick_data = self._data.loc[tick]
        if tick_data.empty or self.last_tick is None:
            self.last_tick = tick
            return

        last_tick_data = self._data.loc[self.last_tick]
        self.grids = self.get_grids(tick, tick_data)

        vol = self._data.loc[tick][['vol_5', 'ticker']].reset_index(drop=True).set_index('ticker').vol_5
        # split into 5 groups based on signal
        vol = pd.qcut(vol, 5, labels=False, duplicates='drop')
        # buy the top 10% stocks
        target_stocks = vol[vol == 4].index.tolist()

        crt_target_data = tick_data.loc[tick_data.ticker.isin(target_stocks)]
        last_target_data = last_tick_data.loc[last_tick_data.ticker.isin(target_stocks)]
        signal = self.gen_signal(crt_target_data, last_target_data)
        buy_stocks = signal.loc[signal.signal >= 1].sort_values('signal', ascending=False).index
        sell_stocks = signal.loc[signal.signal <= -1].sort_values('signal').index

        # sell first and then buy
        for stock in sell_stocks:
            self.sell(stock, 100) # * abs(signal.loc[stock].signal))

        for stock in buy_stocks:
            self.buy(stock, 100) # * abs(signal.loc[stock].signal))
        self.last_tick = tick

    def risk_manager(self):
        # Calculate the current portfolio value based on today's prices
        zero_count = 0
        for symbol, position in self._broker.position.items():
            if position == [0, 0]:
                zero_count += 1
            else:
                break
        if zero_count == len(self._broker.position):
            return False

        current_portfolio_value = self._broker.market_value
        # Calculate the drawdown
        drawdown = (self._broker.initial_cash - current_portfolio_value) / self._broker.initial_cash

        if drawdown > self.max_drawdown:
            # If drawdown exceeds the acceptable level, liquidate the portfolio
            for stock, pos in self._broker.position.items():
                if pos[0] > 0:
                    self.sell(stock)
            return True

        return False

