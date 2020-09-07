learning_rate = 0.001
batch_size = 32
max_epochs = 40
lookbackwindow = 3
T = lookbackwindow+1


train_portion = 0.7 

use_columns = ['open', 'high','low','close']

predict_target = 'close'

n_step_data = 1  # non-overlapping walk length

# args for TCN model
output_size = 1 # regression task
hidden_layer_sizes = [32,32]
# kernel size should be less than T-1 
kernel_size = 2
dropout = 0.1
weight_decay = 0.1

