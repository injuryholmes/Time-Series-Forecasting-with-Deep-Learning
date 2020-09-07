import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from IPython import embed

def read_file(file_location):
    series = []
    ids = []
    with open(file_location, 'r') as file:
        data = file.read().split("\n")

    for i in range(1, len(data) - 1):
    # for i in range(1, len(data)):
        row = data[i].replace('"', '').split(',')
        series.append(np.array([float(j) for j in row[1:] if j != ""]))
        ids.append(row[0])
    series = np.array(series)
    return series


# 这一步为什么不把train也return出来，这样不是没有改变train的值吗； 不用！PY是pass by reference的
def create_val_set(train, output_size):
    val = []
    for i in range(len(train)):
        val.append(train[i][-output_size:])
        train[i] = train[i][:-output_size]
    return np.array(val)


def chop_series(train, chop_val):
    # CREATE MASK FOR VALUES TO BE CHOPPED
    train_len_mask = [True if len(i) >= chop_val else False for i in train]
    # FILTER AND CHOP TRAIN
    # get the items starting from -chop_val to the end; why does it ignore the values from start to chop_val ? QUESTION HERE
    train = [train[i][-chop_val:] for i in range(len(train)) if train_len_mask[i]]
    return train, train_len_mask




def create_datasets(train_file_location, test_file_location, output_size):
    """
    Return train set; validation set and test set

    @rtype:   numpy.ndarray
    @return:  train.shape=(23000,) val.shape=test.shape=(23000,6) 
    """
    train = read_file(train_file_location)
    test = read_file(test_file_location)
    val = create_val_set(train, output_size)

    return train, val, test


class SeriesDataset(Dataset):


    def __init__(self, dataTrain, dataVal, dataTest, info, variable, chop_value, device):
        """
        category: includes: Micro / Finance / Industry etc.
        get_dummies(): Convert categorical variable into dummy/indicator variables
        @param variable: Yearly/Monthly/Weekly
        """
        dataTrain, mask = chop_series(dataTrain, chop_value)
        self.dataInfoCatOHE = pd.get_dummies(info[info['SP'] == variable]['category']) # (23000, 6)
        self.dataInfoCatHeaders = np.array([i for i in self.dataInfoCatOHE.columns.values]) # (6,) array(['Demographic', 'Finance', 'Industry', 'Macro', 'Micro', 'Other'], dtype='<U11')
        self.dataInfoCat = torch.from_numpy(self.dataInfoCatOHE[mask].values).float() # torch.Size([9060, 6])
        self.dataTrain = [torch.tensor(dataTrain[i]) for i in range(len(dataTrain))]  # ALREADY MASKED IN CHOP FUNCTION
        self.dataVal = [torch.tensor(dataVal[i]) for i in range(len(dataVal)) if mask[i]] # len = 9060
        self.dataTest = [torch.tensor(dataTest[i]) for i in range(len(dataTest)) if mask[i]] # len = 9060
        self.device = device

    def __len__(self):
        return len(self.dataTrain)

    def __getitem__(self, idx):
        return self.dataTrain[idx].to(self.device), \
               self.dataVal[idx].to(self.device), \
               self.dataTest[idx].to(self.device), \
               self.dataInfoCat[idx].to(self.device), \
               idx


def collate_lines(seq_list):
    train_, val_, test_, info_cat_, idx_ = zip(*seq_list)
    train_lens = [len(seq) for seq in train_]
    seq_order = sorted(range(len(train_lens)), key=train_lens.__getitem__, reverse=True)
    train = [train_[i] for i in seq_order]
    val = [val_[i] for i in seq_order]
    test = [test_[i] for i in seq_order]
    info_cat = [info_cat_[i] for i in seq_order]
    idx = [idx_[i] for i in seq_order]
    return train, val, test, info_cat, idx

