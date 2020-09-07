import os

import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from IPython import embed

class SP500(Dataset):
    def __init__(self, folder_dataset, T=20, symbol='AAPL', use_columns=['date', 'open', 'high','low','close'], target='close', start_date='2012-01-01', end_date='2015-12-31', step=1):
        
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.use_columns = use_columns
        self.target = target
        self.T = T
        # Create output dataframe
        self.dates = pd.date_range(self.start_date, self.end_date)
        self.df_data = pd.DataFrame(index=self.dates)
        # breakpoint()
        fn = folder_dataset + "/" + symbol + "_data.csv"

        indexs = [col for col in self.use_columns[1:]]
        indexs.append('Target')

        if (target in self.use_columns):
            df_current = pd.read_csv(fn, index_col='date', usecols=self.use_columns, na_values='nan', parse_dates=True)
            # df_current['Target'] = df_current[self.target]
        else:

            self.use_columns.insert(len(self.use_columns), target)
            df_current = pd.read_csv(fn, index_col='date', usecols=self.use_columns, na_values='nan', parse_dates=True)
            # df_current.rename(columns={target:'Target'}, inplace=True)

        tmp = ['Close_t-1','Close_t-Close_t-1', 'Close_t-Open_t', 'Open_t-Close_t-1', 'Open_t+1-Close_t','Target']
        # breakpoint()
        self.df_data = self.df_data.join(df_current)
        self.df_data = self.df_data.dropna() # excludes weekends
        # breakpoint()
        self.df_data['Close_t-1'] = self.df_data['close']
        self.df_data['Close_t-1'] = self.df_data['Close_t-1'].shift(periods=1,axis="rows")
        # breakpoint()
        self.df_data['Close_t-Close_t-1'] = self.df_data['close'].diff()
        self.df_data[['Close_t-Close_t-1']] = self.df_data[['Close_t-Close_t-1']].fillna(value=0)
        # breakpoint()

        self.df_data['Close_t-Open_t'] = self.df_data['close']-self.df_data['open']
        self.df_data['Open_t-Close_t-1'] = self.df_data['open']-self.df_data['close']+self.df_data['Close_t-Close_t-1']
        self.df_data['Open_t+1-Close_t'] = self.df_data['Open_t-Close_t-1'].shift(periods=-1,axis='rows')
        # breakpoint()
        self.df_data.fillna(method='ffill', inplace=True)
        self.df_data.fillna(method='bfill', inplace=True)
        # breakpoint()


        self.df_data['Target'] = self.df_data['Close_t-Close_t-1']
        self.df_data = self.df_data.drop(columns=['open', 'high','low','close'])
        # breakpoint()
        self.numpy_data = self.df_data[tmp].values

        # self.chunks = torch.FloatTensor(self.train_data).unfold(0, self.T, step).permute(0, 2, 1)
        self.chunks = torch.FloatTensor(self.numpy_data).unfold(0, self.T, step).permute(0, 2, 1)


    def __getitem__(self, index):
        close_t_1 = self.chunks[index,-1 , 0] # yesterday's close price
        x = self.chunks[index, :-1, -5:-1] # delta features
        y = self.chunks[index, -1, -1] #  price to predict 
        return close_t_1, x, y


    def __len__(self):
        return self.chunks.size(0)


class SP500Multistep(Dataset):
    def __init__(self, folder_dataset, symbols=['AAPL'], use_columns=['Date', 'Close'], start_date='2012-01-01',
                 end_date='2015-12-31', step=1, n_in=10, n_out=5):
        """

        :param folder_dataset: str
        :param symbols: list of str
        :param use_columns: bool
        :param start_date: str, date format YYY-MM-DD
        :param end_date: str, date format YYY-MM-DD
        """
        self.scaler = MinMaxScaler()
        self.symbols = symbols
        if len(symbols)==0:
            print("No Symbol was specified")
            return
        self.start_date = start_date
        if len(start_date)==0:
            print("No start date was specified")
            return
        self.end_date = end_date
        if len(end_date)==0:
            print("No end date was specified")
            return
        self.use_columns = use_columns
        if len(use_columns)==0:
            print("No column was specified")
            return

        # Create output dataframe
        self.dates = pd.date_range(self.start_date, self.end_date)
        self.df_data = pd.DataFrame(index=self.dates)

        # Read csv files corresponding to symbols
        for symbol in symbols:
            fn = folder_dataset + "/" + symbol + "_data.csv"
            print(fn)
            df_current = pd.read_csv(fn, index_col='Date', usecols=self.use_columns, na_values='nan', parse_dates=True)
            df_current = df_current.rename(columns={'Close': symbol})
            self.df_data = self.df_data.join(df_current)

        # Replace NaN values with forward then backward filling
        self.df_data.fillna(method='ffill', inplace=True, axis=0)
        self.df_data.fillna(method='bfill', inplace=True, axis=0)
        self.numpy_data = self.df_data.as_matrix(columns=self.symbols)
        self.train_data = self.scaler.fit_transform(self.numpy_data)

        self.chunks = []
        self.chunks_data = torch.FloatTensor(self.train_data).unfold(0, n_in+n_out, step)
        k = 0
        while k < self.chunks_data.size(0):
            self.chunks.append([self.chunks_data[k, :, :n_in], self.chunks_data[k, :, n_in:]])
            k += 1

    def __getitem__(self, index):
        x = torch.FloatTensor(self.chunks[index][0])
        y = torch.FloatTensor(self.chunks[index][1])
        return x, y

    def __len__(self):
        return len(self.chunks)

