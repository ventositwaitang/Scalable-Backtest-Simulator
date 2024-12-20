from utils.utils import assert_msg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle


class Broker:
    def __init__(self, data_loader, cash, commission):
        assert_msg(0 < cash, "Enter the initial cash quantity.：{}".format(cash))
        assert_msg(0 <= commission <= 0.05, "Please input the commission fee rate.：{}".format(commission))
        self.data_loader = data_loader
        self._initial_cash = cash
        self._data = data_loader.data.reset_index()[['date', 'ticker', 'last']]
        self._stocks = self._data['ticker'].unique()
        self._commission = commission
        self._position = {i: [0, 0.0] for i in self._stocks}
        self._pos_history = {i: [] for i in self._stocks}
        self.tick_data = {i: 0.0 for i in self._stocks}
        self._cash = cash
        self._i = 0
        self._last_i = None
        self.day_value = []
        self.maxdown_point = []

    @property
    def cash(self):
        """
        :return: Return the current amount of cash in the account
        """
        return self._cash

    @property
    def position(self):
        """
        :return: Retrieve current account position
        """
        return self._position

    @property
    def initial_cash(self):
        """
        :return: Return initial cash amount
        """
        return self._initial_cash

    @property
    def market_value(self):
        """
        :return: Return current market value
        """
        val = self._cash
        for s in self._stocks:
            pos = self._position.get(s)
            if pos is not None:
                val += pos[0] * pos[1]
        return val

    @property
    def current_price(self):
        """
        :return: Return current market price
        """
        new_tick_data = self._data.loc[self._data['date'] == self._i][['ticker', 'last']].to_dict(orient='records')
        for i in new_tick_data:
            self.tick_data[i['ticker']] = i['last']
        return self.tick_data

    def buy(self, stock, volume):
        """
        Buy all at market price using the remaining funds in the current account.
        """
        assert_msg(stock in self._stocks, "Please enter the stock code.：{}".format(stock))
        if volume < 1:
            return
        amount = float(volume * self.current_price[stock] * (1 + self._commission))
        if amount > self._cash:
            return
        self._cash -= amount

        self._position[stock][0] += volume
        self._position[stock][1] = self.current_price[stock]
        self._pos_history[stock].append([self._i, 'buy', volume, self.current_price[stock]])

    def sell(self, stock, volume=None):
        """
        Sell the remaining positions in the current account
        """
        if volume is None:
            volume = self._position[stock][0]
        if volume == 0:
            return
        assert_msg(stock in self._stocks, "Please enter the correct stock code.：{}".format(stock))
        self._cash += float(min(self._position[stock][0], volume) * self.current_price[stock])
        if self._position[stock][0] <= volume:
            self._position[stock] = [0, 0.0]
        else:
            self._position[stock][0] -= volume
        self._pos_history[stock].append([self._i, 'sell', min(self._position[stock][0], volume), self.current_price[stock]])

    def next(self, tick):
        self._i = tick

    # record daily value
    def write_ratio(self, tick):
        the_value = self.market_value
        self.day_value.append({'date': tick, 'value': the_value})

    def get_absolute_return(self):
        _cash = self.market_value
        return (_cash - self._initial_cash) / self._initial_cash

    # annual return rate
    def get_annualized_return(self):
        _cash = self.market_value
        ratio = (_cash - self._initial_cash) / self._initial_cash
        return_value = (1 + ratio) ** (252 / len(self.day_value)) - 1
        return return_value

    # sharpe ratio
    def get_sharpe_ratio(self):
        _cash = pd.DataFrame(self.day_value)['value']
        daily_riskless_rt = (1 + 0.02)**(1/252) - 1
        daily_return = _cash.pct_change().dropna()
        profit = daily_return.mean()
        risk = daily_return.std()
        sharpe = (profit - daily_riskless_rt)/ risk * (252 ** 0.5)
        return sharpe

        # 最大回撤
    def get_maxdown(self):
        _df = pd.DataFrame(self.day_value)
        return_list = _df['value'].dropna()
        i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list)) + 1
        if i == 0:
            return 0
        j = np.argmax(return_list[:i]) + 1

        # record the maxmum drawdown points
        self.maxdown_point.append(_df.iloc[j])
        self.maxdown_point.append(_df.iloc[i])

        return (return_list[j] - return_list[i]) / (return_list[j])

    def get_result(self):
        _dic = {'return': round(self.get_absolute_return(), 4),
                'annual_rate': round(self.get_annualized_return(), 4),
                'max_drawdown': round(self.get_maxdown(), 4),
                'sharp_ratio': round(self.get_sharpe_ratio(), 4)}
        self.result = _dic
        return _dic

    def save_result(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(path + '/trade_history.pkl', 'wb') as f:
            pickle.dump(self._pos_history, f)
        with open(path + '/result.pkl', 'wb') as f:
            pickle.dump(self.get_result(), f)

        day_value = pd.DataFrame(self.day_value)
        day_value.to_csv(path + '/daily_returns.csv')

    def plot_ratio(self, path, w=20, h=7):
        sns.set()
        _day_price = pd.DataFrame(self.day_value).dropna()
        _day_price['trade_ratio'] = (_day_price['value'] - self._initial_cash) / self._initial_cash

        plt.figure(figsize=(w, h))
        # return ratio
        plt.plot(_day_price['date'], _day_price['trade_ratio'], linewidth='2', color='#1E90FF')
        # set x ticks to be readable
        plt.xticks(_day_price['date'][::int(len(_day_price) / 10)], rotation=45)

        # max drawdown points
        x_list = [date['date'] for date in self.maxdown_point]
        y_list = [(date['value'] - self._initial_cash) / self._initial_cash for date in self.maxdown_point]
        plt.scatter(x_list, y_list, c='g', linewidths=7, marker='o')

        # benchmark
        plt.title('Total Returns {0} | Annualized Returns {1} | Max Drawdown {2} | Sharpe Ratio {3}'.format(
            self.result['return'],
            self.result['annual_rate'],
            self.result['max_drawdown'],
            self.result['sharp_ratio']), fontsize=16)
        plt.grid(True)
        path = path + '/daily_returns.png'
        plt.savefig(path)
        plt.show()










