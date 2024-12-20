from lgbm_strategy import MyLGBMStrategy
from src.broker import Broker
from src.backtest import Backtest
from data_processor import DataLoader1
from utils.utils import assert_msg
import pandas as pd
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')


def main():
    # Load historical data
    data_loader = DataLoader1()
    # Initialize backtesting object
    broker = Broker(data_loader, cash=10000000, commission=0.0001)
    strategy = MyLGBMStrategy(broker, data_loader)
    ret = Backtest(data_loader, strategy, broker, name='lgbm').run()
    print(ret)


if __name__ == '__main__':
    main()