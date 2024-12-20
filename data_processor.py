import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from utils.factors import FactorMaker
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder



def read_file():
    last = pd.read_csv("./data/data_last.csv", dtype={'ticker': str, 'date': str, 'last': float}, parse_dates=['date'])
    mktcap = pd.read_csv('./data/data_mkt_cap.csv', dtype={'ticker': str, 'date': str, 'mktcap': float}, parse_dates=['date'])
    volume = pd.read_csv('./data/data_volume.csv', dtype={'ticker': str, 'date': str, 'volume': float}, parse_dates=['date'])
    sector = pd.read_csv('./data/data_sector.csv', dtype={'ticker': str, 'sector': str})
    return last, mktcap, volume, sector


def data_process(last: pd.DataFrame, mkt_cap: pd.DataFrame, volume: pd.DataFrame):
    def fillna(data):
        ticker = data['ticker']
        data = data.sort_values('date').groupby('ticker', as_index=False).fillna(method='ffill')
        data['ticker'] = ticker
        return data
    # fill NAN by using ffill for each stock id
    if last.isnull().values.any():
        last = fillna(last)
    if volume.isnull().values.any():
        volume = fillna(volume)
    if mkt_cap.isnull().values.any():
        mkt_cap =fillna(mkt_cap)
    last.set_index(['ticker', 'date'], inplace=True)
    volume.set_index(['ticker', 'date'], inplace=True)
    mkt_cap.set_index(['ticker', 'date'], inplace=True)

    # merge data
    data = pd.concat([last, volume, mkt_cap], axis=1).reset_index()

    return data


class DataLoader:
    def __init__(self, standard=True, outliers=True, neutral_sec=True, neutral_mkt=True):
        self.standard = standard
        self.outliers = outliers
        self.neutral_sec = neutral_sec
        self.neutral_mkt = neutral_mkt

        last, mktcap, volume, sector = read_file()
        self.data = data_process(last, mktcap, volume)

        self.factors = pd.DataFrame()
        self.factors['vol_20'] = self.data.set_index('date').groupby('ticker').ewm(span=20).std(numeric_only=True)['last'].rename('vol_20')
        self.factors['vol_5'] = self.data.set_index('date').groupby('ticker').ewm(span=5).std(numeric_only=True)['last'].rename('vol_5')
        self.factors['upper'] = (self.data.set_index('date').groupby('ticker').rolling(window=20, min_periods=2).max()['last'] * 1.05).rename('upper')
        self.factors['lower'] = (self.data.set_index('date').groupby('ticker').rolling(window=20, min_periods=2).min()['last'] * 0.95).rename('lower')


class DataLoader1:
    def __init__(self, standard=True, outliers=True, neutral_sec=True, neutral_mkt=True):
        self.standard = standard
        self.outliers = outliers
        self.neutral_sec = neutral_sec
        self.neutral_mkt = neutral_mkt

        last, mktcap, volume, self.sector = read_file()

        self.data = data_process(last, mktcap, volume)
        factor_maker = FactorMaker(self.data)
        try:
            self.factors = pd.read_csv('./data/factors.csv')
        except:
            self.factors = factor_maker.make_factor().dropna()
        self.factors = self.factors.merge(self.sector, on='ticker', how='left').merge(self.data.reset_index()[['ticker', 'date', 'mkt_cap']], on=['ticker', 'date'], how='left')

    def neutralize(self):
        # One-hot encode the sector information
        self.factors = self.factors.set_index(['ticker', 'date'])
        encoder = OneHotEncoder(sparse=False)
        sector_encoded = encoder.fit_transform(self.factors[['bics_sector']])
        sector_encoded_df = pd.DataFrame(sector_encoded, columns=encoder.get_feature_names_out(['bics_sector']))
        sector_encoded_df.index = self.factors.index
        # Add the market cap to the one-hot encoded DataFrame
        X = pd.concat([np.log(self.factors[['mkt_cap']]), sector_encoded_df], axis=1)
        X = sm.add_constant(X)  # Adds a constant term to the predictor
        neutralized_factors = pd.DataFrame(index=self.factors.index)

        for factor in self.factors.columns.drop(['bics_sector']):
            # Perform the regression and save the residuals
            y = self.factors[factor]
            model = sm.OLS(y, X).fit()
            neutralized_factors[factor] = model.resid

        return neutralized_factors

    def make_data(self, data):
        if self.outliers:
            data = self.outlier(data)
        if self.standard:
            data = self.standardize(data)
        return data

    def make_y(self):
        data = self.data.reset_index()[['ticker', 'date', 'last']].set_index(['ticker', 'date'])
        y = data.groupby(level='ticker').pct_change().shift(-1)
        data['y'] = y
        return data.dropna()['y']

    def make_x(self):
        return self.factors.drop(['bics_sector', 'mkt_cap'], axis=1)

    def make_xy(self):
        x = self.make_x()
        y = self.make_y()
        xy = pd.concat([x, y], axis=1).dropna()
        return xy.reset_index(level=0)

    def _standardize(self, data):
        infos = data[['ticker', 'date', 'bics_sector']]
        standardized_data = data.drop(['ticker', 'date', 'bics_sector'], axis=1).to_numpy()
        new_data = np.zeros(standardized_data.shape)
        for i in range(len(data)):
            window_data = standardized_data[:i + 1]
            window_mean = np.mean(window_data, axis=0)
            window_std = np.std(window_data, axis=0)

            # Standardize the current data point
            new_data[i] = (standardized_data[i] - window_mean) / (window_std + 10e-10)

        standardized_data = pd.DataFrame(new_data, columns=data.columns.drop(infos.columns), index=data.index)
        standardized_data = pd.concat([infos, standardized_data], axis=1)
        return standardized_data

    def standardize(self, data):
        # standardize the data
        data = data.groupby('ticker', as_index=False).apply(self._standardize)
        return data.reset_index(drop=True)

    def _outlier(self, data: pd.DataFrame):
        infos = data[['ticker', 'date', 'bics_sector']]
        outlier_data = data.drop(['ticker', 'date', 'bics_sector'], axis=1).to_numpy()
        new_data = np.zeros(outlier_data.shape)
        for i in range(len(data)):
            window_data = outlier_data[:i + 1]

            window_mean = np.mean(window_data, axis=0)
            window_std = np.std(window_data, axis=0)

            # Standardize the current data point
            new_data[i] = np.clip(outlier_data[i], window_mean - 3 * window_std, window_mean + 3 * window_std)
        outlier_data = pd.DataFrame(new_data, columns=data.columns.drop(['ticker', 'date', 'bics_sector']), index=data.index)
        outlier_data = pd.concat([infos, outlier_data], axis=1)
        return outlier_data

    def outlier(self, data):
        # remove outliers
        data = data.groupby('ticker', as_index=False).apply(self._outlier)
        return data.reset_index(drop=True)





