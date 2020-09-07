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
# Local imports
from utils.data_sp500_delta_lookahead_multiclass import SP500
from TCN.model_bn_tri import TCN
from torch import autograd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import config.stock_config as cfg
import pandas as pd
from IPython import embed
import time
from drawloss import analysis

classes = 3 # number of classes for classification task
# symbol = '^GSPC'
symbol = '^shanghai'

if symbol == '^GSPC':
    train_start = pd.to_datetime("2001-01-01")
    test_end =  pd.to_datetime("2017-05-31")
elif symbol == '^shanghai':
    train_start = pd.to_datetime("2005-01-01")
    test_end =  pd.to_datetime("2017-05-31")

delta_days = test_end - train_start
train_end = train_start + delta_days*0.6
test_start = test_end - delta_days*0.2

train_start = train_start.strftime('%Y-%m-%d')
train_end = train_end.strftime('%Y-%m-%d')
test_start = test_start.strftime('%Y-%m-%d')
test_end = test_end.strftime('%Y-%m-%d')
input_channels = 4 # Three input features: ['Close_t-Close_t-1', Close_t-Open_t', 'Open_t-Close_t-1','Open_t+1-Close_t']
file = pd.read_csv('data/stock/sandp500/individual_stocks_5yr/'+symbol+'_data.csv', index_col='date', parse_dates=True)
dtrain = SP500('data/stock/sandp500/individual_stocks_5yr', symbol=symbol, target=cfg.predict_target, start_date=train_start, end_date=train_end, T=cfg.T, step=cfg.n_step_data)
train_loader = DataLoader(dtrain, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
dtest = SP500('data/stock/sandp500/individual_stocks_5yr', symbol=symbol, target=cfg.predict_target, start_date=test_start, end_date=test_end,T=cfg.T,step=cfg.n_step_data)
test_loader = DataLoader(dtest,batch_size=cfg.batch_size,shuffle=False,num_workers=4)
# Network Definition + Optimizer + Scheduler
model = TCN(input_channels, cfg.output_size, cfg.hidden_layer_sizes, kernel_size=cfg.kernel_size, dropout=cfg.dropout, multiclass=classes)
optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
scheduler_model = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2) # https://kite.com/python/docs/torch.optim.lr_scheduler.StepLR 

criterion = nn.MSELoss(size_average=True)
criterion2 = nn.CrossEntropyLoss()

def MAPELoss(output, target):
    epsilon = 1e-8
    if type(output) == np.ndarray:
        return  np.mean((np.abs(target - output)+epsilon) / (np.abs(target)+epsilon))
    else:
        return torch.mean((torch.abs(target - output)+epsilon) / (torch.abs(target)+epsilon))


def evaluate():
    predictions, closes, gts = [], [], []
    model.eval()
    with torch.no_grad():
        ce_i = []
        for batch_idx, (close_t_1, data, target, multiclass) in enumerate(test_loader):
            data = Variable(data.permute(0, 2, 1)).contiguous() # torch.Size([16, 1, 7, 19])            
            target = Variable(target.unsqueeze_(1))
            triple_target = multiclass.long()
            
            output,tri_out = model(data)
            closes.extend(close_t_1.data)
            predictions.extend(output.data)
            gts.extend(target.data)
            ce = criterion2(tri_out, triple_target)
            ce_i.append(ce)
    pred = np.array(predictions).reshape(-1,1)
    closes = np.array(closes).reshape(-1,1)
    gts = np.array(gts).reshape(-1,1)
    pred = pred + closes
    gts = gts + closes
    mse = mean_squared_error(pred, gts)
    mae = mean_absolute_error(pred, gts)
    r2 = r2_score(pred, gts)
    mape = MAPELoss(pred, gts)
    return mse, mae, r2, mape, np.mean(ce_i)

if __name__ == "__main__":
    mselosses = []
    celosses = []
    mapelosses = []
    test_mses = []
    test_maes=[]
    test_mapes = []
    test_r2s = []
    test_ces= []

    
    model.train()
    start_time = time.time()
    for i in range(cfg.max_epochs):
        loss_1 = []
        loss_2 = []
        loss_3 = []
        # Go through training data set
        for batch_idx, (close, data, target, multiclass) in enumerate(train_loader): # data: torch.Size([16, 19, 7]); target: torch.Size([16, 7])
            data = Variable(data.permute(0, 2, 1)).contiguous()
            triple_target = multiclass.long()
            target = Variable(target.unsqueeze_(1))
            # breakpoint()
            optimizer.zero_grad() # Set gradient of optimizer to 0
            output, tri_out = model(data)       
            loss1 = criterion(output, target) # the loss for all the data pairs within the batch 
            loss2 = criterion2(tri_out, triple_target)
            loss_1.append(loss1.item())# 记录每个epoch整体的loss
            loss_2.append(loss2.item())
            
            loss3 = MAPELoss(output, target)
            loss_3.append(loss3.item())
            # loss = loss2
            # loss = loss1*(loss2**1.2)
            # loss = loss1/50 +loss2 
            # loss = loss1/1000 +loss2 
            loss = loss3
            # loss = loss1*(loss2**2)

            # loss = loss3
            loss.backward()
            optimizer.step() # Gradient descent step

        print("Epoch = ", i)
        print("mean mse Loss = ", np.mean(loss_1))
        print("mean bce loss = ", np.mean(loss_2))
        print("mean mape loss = ", np.mean(loss_3))


        mselosses.append(np.mean(loss_1))
        celosses.append(np.mean(loss_2))
        mapelosses.append(np.mean(loss_3))

        scheduler_model.step() # Apply step of scheduler for learning rate change
        mse, mae, r2, mape, ce  = evaluate()
        test_mses.append(mse)
        test_maes.append(mae)
        test_r2s.append(r2)
        test_mapes.append(mape)
        test_ces.append(ce)
    print("--- %s seconds ---" % (time.time() - start_time))
    fig,ax = plt.subplots()
    x = range(len(mapelosses))
    ax.plot(x, mapelosses, color="red", label="train mape")
    ax.set_xlabel("epoch",fontsize=14)
    ax.set_ylabel("train loss",color="red",fontsize=14)
    plt.legend(loc='upper center')
    
    ax2=ax.twinx()
    ax2.plot(x, test_mapes,color='blue',label="test mape")
    ax2.set_ylabel("test loss",color="blue",fontsize=14)
    plt.legend(loc='upper right')
    plt.savefig("Shanghai lookahead MAPE"+str(cfg.lookbackwindow)+".png",bbox_inches = 'tight')
    plt.show()


    # fig,ax = plt.subplots()
    # x = range(len(celosses))
    # ax.plot(x, celosses, color="red", label="train ce")
    # ax.set_xlabel("epoch",fontsize=14)
    # ax.set_ylabel("train loss",color="red",fontsize=14)
    # plt.legend(loc='upper center')
    
    # ax2=ax.twinx()
    # ax2.plot(x, test_ces,color='blue',label="test ce")
    # ax2.set_ylabel("test loss",color="blue",fontsize=14)
    # plt.legend(loc='upper right')
    # plt.savefig("16TTL ce lookahead.png",bbox_inches = 'tight')
    # plt.show()



    ##########################################################################################
    # TEST
    predictions = []
    closes = []
    gts = []

    model.eval()

    with torch.no_grad():
        ce_i = []
        for batch_idx, (close_t_1, data, target, multiclass) in enumerate(test_loader):
            data = Variable(data.permute(0, 2, 1)).contiguous() # torch.Size([16, 1, 7, 19])            
            target = Variable(target.unsqueeze_(1))
            triple_target = multiclass.long()
            
            output,tri_out = model(data)
            closes.extend(close_t_1.data)
            predictions.extend(output.data)
            gts.extend(target.data)
            ce = criterion2(tri_out, triple_target)
            ce_i.append(ce)
    CE_loss = np.mean(ce_i)
    print("CE_loss: "+str(CE_loss))
    pred = np.array(predictions).reshape(-1,1)
    closes = np.array(closes).reshape(-1,1)
    gts = np.array(gts).reshape(-1,1)

    count = 0
    for i in range(len(pred)):
        if pred[i] * gts[i] >=0 :
            count += 1

    pred = pred + closes
    gts = gts + closes
    loss_time = []
    for i in range(len(pred)):
        loss_time.append((pred[i].item()-gts[i].item())**2)
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

    # breakpoint()
    scaled_bench_loss = [2*np.mean(gts)-s for s in scaled_bench_loss]

    benchmark_mse = mean_squared_error(df['gts_prev'], df['gts']) # mse between the benchmark and the ground truth
    benchmark_mae = mean_absolute_error(df['gts_prev'], df['gts'])
    benchmark_r2 = r2_score(df['gts_prev'], df['gts'])
    df['delta_pred'] = df['pred'] - df['gts_prev']  # the trend predicted by TCN, > 0 means up; <0 means down
    df['delta_benchmark'] = df['gts_prev'] - df['gts'] # the trend predicted by benchmark, > 0 means up; <0 means down
    df['same_sign_benchmark'] = df['delta_gts'] * df['delta_benchmark'] # the accuracy of the benchmark, which are up up or down down.
    benchmark_accuracy =  len(df.loc[df.same_sign_benchmark >= 0]) # number of correct predictions of benchmark
    df['same_sign_tcn'] = df['delta_gts'] * df['delta_pred'] # the accuracy of the tcn, which are up up or down down.
    tcn_accuracy = len(df.loc[df.same_sign_tcn >= 0]) # number of correct predictions of TCN
    df['delta_gts_ratio'] = df['delta_gts'].abs()/df['gts_prev']
    df['rmse_pred_gt'] = df['pred']-df['gts']
    df['rmse_pred_gt'] = df.rmse_pred_gt.abs()
    # ss = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,]
    # lst = list(range(250))
    all_rmse =  df['rmse_pred_gt'].to_numpy().tolist()
    all_rmse = [round(ele,2) for ele in all_rmse]
    # calc_win_rmse_ratio(all_rmse)
    if symbol == '^GSPC':
        lst = list(range(800))
    else:
        lst = list(range(500))
    ss = lst[5::5]
    ts = []
    ress = []
    for i in ss:
        top = round(sum(df.nlargest(i,'delta_gts_ratio').rmse_pred_gt),2)
        total = round(sum(df.rmse_pred_gt) - top,2)
        ts.append(top)
        ress.append(total)
    # print("all rmse: " + str(all_rmse))
    # print("ts: "+ str(ts))
    # print("ress: " + str(ress))
    analysis(symbol, all_rmse, ts, ress)
    print("================================run_stock_raw_tcn_delta")

    # print("pred binary accuracy: " + str(tcn_accuracy/len(pred)))
    # print("benchmark binary accuracy: " + str(benchmark_accuracy/len(pred)))
    mse = mean_squared_error(pred, gts)
    mae = mean_absolute_error(pred, gts)
    r2 = r2_score(pred, gts)
    print("number of test data: " + str(len(pred)))
    print("MSE: " + str(mse))
    print("benchmark mse: " + str(benchmark_mse))
    print("benchmark MAE: " + str(benchmark_mae))
    print("r2: " + str(r2))
    print("benchmark_r2: "+str(benchmark_r2))
    print("benchmark rmse: " + str(sqrt(benchmark_mse)))

    print("RMSE: " + str(sqrt(mse)))
    print("MAE: " + str(mae))
    print("win ratio: " + str(win/len(delta_loss)))
    print("classification accuracy: " + str(count / len(pred)));
        
    # Plot for all stocks in
    x = [np.datetime64(test_start) + np.timedelta64(x, 'D') for x in range(0, pred.shape[0])]
    x = np.array(x)
    months = MonthLocator(range(1, 10), bymonthday=1, interval=3)
    monthsFmt = DateFormatter("%b '%y")

    # for stock in symbols:
    # fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    # plt.plot(x, pred[:], label="predictions", color=cm.Blues(300))
    # plt.plot(x, gts[:], label="true", color=cm.Blues(100))
    # plt.plot(x, scaled_loss[:], label="loss_time", color=cm.Reds(50))
    # plt.plot(x, scaled_bench_loss[:], label="bench_loss", color=cm.Greens(50))
    # plt.plot(x, delta_loss[:], label="delta_loss", color='black')

    # ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    # ax.xaxis.set_major_locator(months)
    # ax.xaxis.set_major_formatter(monthsFmt)
    # plt.title(symbol)
    # plt.xlabel("Time")
    # plt.ylabel("Stock Price")
    # plt.legend()
    # fig.autofmt_xdate()
    # plt.savefig('16c.png')
    # plt.show()
