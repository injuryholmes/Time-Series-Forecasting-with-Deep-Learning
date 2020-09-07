import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter
from utils.data_sp500_minmax import SP500
from TCN.model import TCN
from models.lstm import LSTM
from IPython import embed
from torch import autograd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import config.stock_config as cfg
import pandas as pd
import time
def MAPELoss(output, target):
    epsilon = 1e-10
    return 100*torch.mean(torch.abs((target - output)) / (1+target+epsilon))


def evaluate():
    predictions = []
    gts = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = Variable(data.permute(0, 2, 1)).contiguous() # torch.Size([16, 1, 7, 19])            
            target = Variable(target.unsqueeze_(1))
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            predictions.extend(output.data)
            gts.extend(target.data)

    pred = np.array(predictions).reshape(-1,1)
    for i in range(len(use_columns)-1):
        pred = np.hstack((pred, pred[:, [-1]]))

    gts = np.array(gts).reshape(-1,1)
    for i in range(len(use_columns)-1):
        gts = np.hstack((gts, gts[:, [-1]]))
    pred = dtest.scaler.inverse_transform(pred)[:,-1].reshape(-1,1)
    gts = dtest.scaler.inverse_transform(gts)[:, -1].reshape(-1,1)

    mse = mean_squared_error(pred, gts)
    mae = mean_absolute_error(pred, gts)
    r2 = r2_score(pred, gts)
    return mse, mae, r2


# symbol = '^GSPC'
# train_start = pd.to_datetime("2001-01-01")
# test_end =  pd.to_datetime("2017-05-31")
# delta_days = test_end - train_start
# train_end = train_start +delta_days*0.6
# test_start = test_end - delta_days*0.2
##### This block specifies the start and end dates of Shanghai index used in another published paper

symbol = '^shanghai'
train_start = pd.to_datetime("2005-01-01")
test_end =  pd.to_datetime("2017-05-31")
delta_days = test_end - train_start
train_end = train_start +delta_days*0.6
test_start = test_end - delta_days*0.2

train_start = train_start.strftime('%Y-%m-%d')
train_end = train_end.strftime('%Y-%m-%d')
test_start = test_start.strftime('%Y-%m-%d')
test_end = test_end.strftime('%Y-%m-%d')

use_cuda = torch.cuda.is_available()
use_columns = ['date', 'open', 'high','low','close']
input_channels = 4 
file = pd.read_csv('data/stock/sandp500/individual_stocks_5yr/'+symbol+'_data.csv', index_col='date', parse_dates=True)
  
# Training data
dtrain = SP500('data/stock/sandp500/individual_stocks_5yr', symbol=symbol, 
    use_columns=use_columns.copy(), target=cfg.predict_target, 
    start_date=train_start, end_date=train_end, T=cfg.T, step=cfg.n_step_data)
train_loader = DataLoader(dtrain, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=use_cuda)

dtest = SP500('data/stock/sandp500/individual_stocks_5yr', symbol=symbol, 
    use_columns=use_columns.copy(), target=cfg.predict_target, 
    start_date=test_start, end_date=test_end,T=cfg.T,step=cfg.n_step_data)
test_loader = DataLoader(dtest,batch_size=cfg.batch_size,shuffle=False,num_workers=4,pin_memory=use_cuda)

model = LSTM(hidden_size=128, num_securities=input_channels, dropout=cfg.dropout, n_layers=2, T=cfg.T)

optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=0.0)
scheduler_model = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9) # https://kite.com/python/docs/torch.optim.lr_scheduler.StepLR 

if __name__ == "__main__":
    criterion = nn.MSELoss(size_average=True)
    # Store successive losses
    losses = []
    test_mses = []
    test_maes=[]
    test_r2s = []
    
    model.train()
    start_time = time.time()
    for i in range(cfg.max_epochs):
        loss_epoch = []
        for batch_idx, (data, target) in enumerate(train_loader): # data: torch.Size([16, 19, 7]); target: torch.Size([16, 7])
            data = Variable(data.permute(0, 2, 1)).contiguous()
            target = Variable(target.unsqueeze_(1))
            optimizer.zero_grad() # Set gradient of optimizer to 0
            output = model(data)            
            loss = criterion(output, target) # the loss for all the data pairs within the batch 
            loss_epoch.append(loss.item()) # 记录每个epoch整体的loss

            loss.backward()
            optimizer.step() # Gradient descent step

        print("Epoch = ", i)
        print("Loss = ", np.mean(loss_epoch))
        losses.append(np.mean(loss_epoch))
        scheduler_model.step()
        mse, mae, r2  = evaluate()
        test_mses.append(mse)
        test_maes.append(mae)
        test_r2s.append(r2)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    h = plt.figure()
    x = range(len(losses))
    plt.plot(np.array(x), np.array(losses), label="train loss",color=cm.Blues(100))
    plt.plot(np.array(x), test_mses, label="test mse",color='green')
    plt.xlabel("Time")
    plt.ylabel("Training loss")
    plt.legend()
    plt.show()

    ##########################################################################################
    # TEST

    predictions = []
    gts = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            print(batch_idx)
            data = Variable(data.permute(0, 2, 1)).contiguous() # torch.Size([16, 1, 7, 19])            
            target = Variable(target.unsqueeze_(1))
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            predictions.extend(output.data)
            gts.extend(target.data)

    # Plot results
    # Convert lists to np array for plot, and rescaling to original data
    # breakpoint()
    pred = np.array(predictions).reshape(-1,1)
    for i in range(len(use_columns)-1):
        pred = np.hstack((pred, pred[:, [-1]]))

    gts = np.array(gts).reshape(-1,1)
    for i in range(len(use_columns)-1):
        gts = np.hstack((gts, gts[:, [-1]]))
    # breakpoint()
    pred = dtest.scaler.inverse_transform(pred)[:,-1].reshape(-1,1)
    gts = dtest.scaler.inverse_transform(gts)[:, -1].reshape(-1,1)

    loss_time = [] # calculate the rmse for at each data point
    for i in range(len(pred)):
        loss_time.append((pred[i].item()-gts[i].item())**2)

    # rescale the loss_time into [mean(gts), max(gts)], this is for the convenience of visualization
    scaled_loss = np.interp(loss_time, (np.min(loss_time), np.max(loss_time)), (np.mean(gts), np.max(gts)))
    

    cols = np.concatenate((pred, gts),axis=1)

    df = pd.DataFrame(cols,columns=['pred','gts'])

    df['delta_gts'] = df['gts'].diff() # delta_gts: close price today - close price yesterday
    df = df.fillna(0)
    df['gts_prev'] = df['gts'].shift(periods=1,axis='rows') # gts_prev: close price yesterday; we assume the benchmark prediction always predict yesterday's close value as today's price
    df.fillna(method='bfill', inplace=True, axis=0)
    
    bench_loss_time = []
    for i in range(len(pred)):
        bench_loss_time.append((df['gts_prev'][i].item()-df['gts'][i].item())**2)

    scaled_bench_loss = np.interp(bench_loss_time, (np.min(bench_loss_time), np.max(bench_loss_time)), (np.mean(gts), np.max(gts)))
    
    delta_loss = []
    win = 0
    for i in range(len(pred)):
        delta = scaled_loss[i]-scaled_bench_loss[i]
        delta_loss.append(np.mean(gts)+delta)
        if delta < 0:
            win += 1
    # reverse the scaled_bench_loss for the convenience of visualization
    scaled_bench_loss = [2*np.mean(gts)-s for s in scaled_bench_loss]

    
    benchmark_mse = mean_squared_error(df['gts_prev'], df['gts']) # mse between the benchmark and the ground truth
    benchmark_mae = mean_absolute_error(df['gts_prev'], df['gts'])
    benchmark_r2 = r2_score(df['gts_prev'], df['gts'])

    df['delta_pred'] = df['pred'] - df['gts_prev']  # the trend predicted by TCN, > 0 means up; <0 means down
    df['delta_benchmark'] = df['gts_prev'] - df['gts'] # the trend predicted by benchmark, > 0 means up; <0 means down
    df['same_sign_benchmark'] = df['delta_gts'] * df['delta_benchmark'] # the accuracy of the benchmark, which are up up or down down.
    benchmark_accuracy =  len(df.loc[df.same_sign_benchmark > 0]) # number of correct predictions of benchmark
    df['same_sign_tcn'] = df['delta_gts'] * df['delta_pred'] # the accuracy of the tcn, which are up up or down down.
    tcn_accuracy = len(df.loc[df.same_sign_tcn > 0]) # number of correct predictions of TCN


    print("================================run_stock_minmax_lstm")

    print("pred binary accuracy: " + str(tcn_accuracy/len(pred)))
    print("benchmark binary accuracy: " + str(benchmark_accuracy/len(pred)))
    print("win ratio: " + str(win/len(delta_loss))) # number that tcn perform more accuracy than benchmark


    # breakpoint()
    mse = mean_squared_error(pred, gts)
    mae = mean_absolute_error(pred, gts)
    r2 = r2_score(pred, gts)

    mape = MAPELoss(torch.from_numpy(pred), torch.from_numpy(gts))
    mape_bench = MAPELoss(torch.from_numpy(df['gts_prev'].to_numpy()), torch.from_numpy(df['gts'].to_numpy()))
    print("MAPE: " + str(mape.item()))
    print("Benchmark mape: " + str(mape_bench.item()))
    print("MSE: " + str(mse))
    print("benchmark mse: " + str(benchmark_mse))
    print("MAE: " + str(mae))
    print("benchmark MAE: " + str(benchmark_mae))
    print("r2: " + str(r2))
    print("benchmark_r2: "+str(benchmark_r2))
    print("RMSE: " + str(sqrt(mse)))
    print("benchmark rmse: " + str(sqrt(benchmark_mse)))
    # Plot for all stocks in
    x = [np.datetime64(test_start) + np.timedelta64(x, 'D') for x in range(0, pred.shape[0])]
    x = np.array(x)
    months = MonthLocator(range(1, 10), bymonthday=1, interval=3)
    monthsFmt = DateFormatter("%b '%y")

    # for stock in symbols:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    plt.plot(x, pred[:], label="predictions", color=cm.Blues(300))
    plt.plot(x, gts[:], label="true", color=cm.Blues(100))
    plt.plot(x, scaled_loss[:], label="loss_time", color=cm.Reds(50))
    plt.plot(x, scaled_bench_loss[:], label="bench_loss", color=cm.Greens(50))
    plt.plot(x, delta_loss[:], label="delta_loss", color='black')
    
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    plt.title(symbol)
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    fig.autofmt_xdate()
    plt.savefig('minmax_lstm.png')
    plt.show()
