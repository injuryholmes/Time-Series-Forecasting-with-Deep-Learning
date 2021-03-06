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
from models.dilated_cnn import DilatedNet2D
from utils.sp500_data_raw import SP500
from TCN.model_bn import TCN
from torch import autograd
from sklearn.metrics import mean_squared_error
import config.stock_config as cfg

from IPython import embed

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()

    use_columns = cfg.use_columns
    input_channels = len(use_columns)

    use_columns.insert(0, 'Date') # always put Date as the first column

    log = []
    for batch_size in cfg.batch_sizes:
        for T in cfg.Ts:
            for kernel_size in cfg.kernel_sizes:
                fn_base = 'batchsize_' + str(batch_size) + '_T_' + str(T) + '_kernel_' + str(kernel_size) + '_'

                # Training data
                dtrain = SP500('data/stock/sandp500/individual_stocks_5yr', symbol=cfg.symbol, 
                    use_columns=use_columns.copy(), target=cfg.predict_target, 
                    start_date=cfg.train_start, end_date=cfg.train_end, T=T, step=cfg.n_step_data)
                train_loader = DataLoader(dtrain, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=use_cuda)
                
                dtest = SP500('data/stock/sandp500/individual_stocks_5yr', symbol=cfg.symbol, 
                    use_columns=use_columns.copy(), target=cfg.predict_target, 
                    start_date=cfg.test_start, end_date=cfg.test_end,T=T,step=cfg.n_step_data)
                test_loader = DataLoader(dtest,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=use_cuda)

                # Network Definition + Optimizer + Scheduler
                model = TCN(input_channels, cfg.output_size, cfg.hidden_layer_sizes, kernel_size=kernel_size, dropout=cfg.dropout)

                if use_cuda:
                    model = model.cuda()

                optimizer = optim.RMSprop(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
                scheduler_model = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9) # https://kite.com/python/docs/torch.optim.lr_scheduler.StepLR 

                # input_sample, _ = next(iter(train_loader))
                # input_sample = Variable(input_sample.permute(0, 2, 1)).unsqueeze_(1).contiguous()
                # writer.add_graph(model, input_sample)

                criterion = nn.MSELoss(size_average=True).cuda()
                # Store successive losses
                losses = []
                
                model.train()
                # with autograd.detect_anomaly():
                for i in range(cfg.max_epochs):
                    loss_epoch = 0.
                    # Go through training data set
                    for batch_idx, (data, target) in enumerate(train_loader): # data: torch.Size([16, 19, 7]); target: torch.Size([16, 7])
                        data = Variable(data.permute(0, 2, 1)).contiguous()
                        target = Variable(target.unsqueeze_(1))
                        if use_cuda:
                            data = data.cuda()
                            target = target.cuda()
                            
                        optimizer.zero_grad() # Set gradient of optimizer to 0
                        # breakpoint()

                        output = model(data)            
                        loss = criterion(output, target) # the loss for all the data pairs within the batch 
                        if (np.isnan(loss.item())):
                            print(fn_base)
                            exit(0)
                        #     embed()
                        loss_epoch += loss.item() # 记录每个epoch整体的loss

                        loss.backward()
                        optimizer.step() # Gradient descent step

                    print("Epoch = ", i)
                    print("Loss = ", loss_epoch)
                    losses.append(loss_epoch)
                    # writer.add_scalar("loss_epoch", loss_epoch, i) # 显示每个epoch的loss
                    scheduler_model.step() # Apply step of scheduler for learning rate change
                # writer.close()

                # Save trained models
                # torch.save(model, 'conv2d_' + fn_base + '.pkl')
                # Plot training loss
                # h = plt.figure()
                # x = range(len(losses))
                
                # plt.plot(np.array(x), np.array(losses), label="loss")
                # plt.xlabel("Time")
                # plt.ylabel("Training loss")
                # plt.savefig("loss_" + fn_base + '.png')
                # plt.legend()
                # plt.show()

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

                pred = np.array(predictions).reshape(-1,1)

                gts = np.array(gts).reshape(-1,1)

                mse = mean_squared_error(pred, gts)

                print("MSE: " + str(mse))
                # Plot for all stocks in
                x = [np.datetime64(cfg.test_start) + np.timedelta64(x, 'D') for x in range(0, pred.shape[0])]
                x = np.array(x)
                months = MonthLocator(range(1, 10), bymonthday=1, interval=3)
                monthsFmt = DateFormatter("%b '%y")

                # for stock in symbols:
                # fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
                # plt.plot(x, pred[:], label="predictions", color=cm.Blues(300))
                # plt.plot(x, gts[:], label="true", color=cm.Blues(100))
                # ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
                # ax.xaxis.set_major_locator(months)
                # ax.xaxis.set_major_formatter(monthsFmt)
                # plt.title(cfg.symbol)
                # plt.xlabel("Time")
                # plt.ylabel("Stock Price")
                # plt.legend()
                # fig.autofmt_xdate()
                # plt.savefig("raw " + fn_base + "mse_" + str(mse)+ "use_columns: " + "_".join(str(x) for x in use_columns) + '.png')
                # plt.show()
                log.append(fn_base + "mse_" + str(mse))
    print(*log,sep="\n")
