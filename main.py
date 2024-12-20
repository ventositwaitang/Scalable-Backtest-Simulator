from mystrategy import MyStrategy
from src.broker import Broker
from src.backtest import Backtest
from data_processor import DataLoader
from utils.utils import assert_msg
import pandas as pd
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')


def main():
    # Load historical data
    data_loader = DataLoader()

    # Initialize backtesting object
    broker = Broker(data_loader, cash=10000000, commission=0.0001)
    strategy = MyStrategy(broker, risk_manage=True, model='ridge')
    ret = Backtest(data_loader, strategy, broker, name='easy').run()
    print(ret)


if __name__ == '__main__':
    main()