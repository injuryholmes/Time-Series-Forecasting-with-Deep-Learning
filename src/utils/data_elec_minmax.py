from torch.utils.data.dataset import Dataset

from torch.utils.data import DataLoader
import torch

import pandas as pd
from sklearn.preprocessing import MinMaxScaler




def data_prep(data, n_in=1, n_out=1, dropnan=True):
        names, cols = list(), list()
        data_frame = pd.DataFrame(data)
        n = 1 if type(data) is list else data.shape[1]
        for x in range(n_in, 0, -1):  #  Input Sequence (t-n, ... t-1)
            cols.append(data_frame.shift(x))
            names += [('var%d(t-%d)' % (y+1, x)) for y in range(n)]
    #     breakpoint()
        
        for x in range(0, n_out):  #  Forecast Sequence (t, t+1, ... t+n)
            cols.append(data_frame.shift(-x))
            if x == 0:
                names += [('var%d(t)' % (y+1)) for y in range(n)]
            else:
                names += [('var%d(t+%d)' % (y+1, x)) for y in range(n)]
        
        z = pd.concat(cols, axis=1)  #  Putting It All Together
        z.columns = names
        
        if dropnan:  #  Dropping Rows With NaN Values
            z.dropna(inplace=True)
        return z

class Elec(Dataset):
    def __init__(self, horizon='D', data_type='test', use_columns=['Global_reactive_power', 'Sub_metering_3'], target='Global_active_power', train_portion=0.6, T=4, step=2):
        total_cols = ['Global_active_power', 'Global_reactive_power', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        self.use_columns = use_columns
        self.scaler = MinMaxScaler()

        power = pd.read_csv('/Users/allenholmes/Desktop/Time-Series-Forecasting-with-Deep-Learning/data/UCI/household_power_consumption.txt', sep=';', 
                 parse_dates={'Date_Time' : ['Date', 'Time']}, infer_datetime_format=True, 
                 low_memory=False, na_values=['nan','?'], index_col='Date_Time')

        power.fillna(method='ffill', inplace=True)
        power.fillna(method='bfill', inplace=True)

        power.drop(['Voltage'],1,inplace=True)
        power['Target'] = power[target]

        # breakpoint()
        #  Dropping Feature: Voltage
        for col in total_cols:
            if not col in self.use_columns:
                # print(col)
                power.drop(col,1,inplace=True)

        if horizon == 'T':
            resample_data = power.resample('T').mean() # min
        elif horizon == 'H':
            resample_data = power.resample('H').mean() # hour
        elif horizon == 'D':
            resample_data = power.resample('D').mean() # day
        elif horizon == 'W':
            resample_data = power.resample('W').mean() # week
        elif horizon == 'M':
            resample_data = power.resample('M').mean() # month
        elif horizon == 'Q':
            resample_data = power.resample('Q').mean() # quarter


        raw = resample_data.values
        duration = int(resample_data.shape[0] * train_portion)               


        if data_type == 'train':
            train = raw[:duration, :]
            minmax_ = self.scaler.fit_transform(train)
            self.chunks = torch.FloatTensor(minmax_).unfold(0, T, step).permute(0, 2, 1)
        elif data_type == 'test':
            test = raw[duration:, :]
            # breakpoint()
            minmax_ = self.scaler.fit_transform(test)
            self.chunks = torch.FloatTensor(minmax_).unfold(0, T, step).permute(0, 2, 1)


    def __getitem__(self, index):
        x = self.chunks[index, :-1, :-1] # previous features
        y = self.chunks[index, -1, -1] # tomorrow's price
        return x, y

    def __len__(self):
        return self.chunks.size(0)




if __name__ == "__main__":
    dset = Elec(horizon='D', data_type='test', train_portion=0.6)
    train_loader = DataLoader(dset, batch_size=16, shuffle=True, num_workers=4, pin_memory=False)

    for i in range(4):
        loss_epoch = 0.
        # Go through training data set
        for batch_idx, (data, target) in enumerate(train_loader): # data: torch.Size([16, 19, 7]); target: torch.Size([16, 7])
            pass
            # breakpoint()
        print("Epoch = ", i)

