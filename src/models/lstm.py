"""
Project:    stock_prediction
File:       lstm.py
Created by: louise
On:         08/02/18
At:         12:55 PM
"""
import torch
from torch import nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, hidden_size=64, num_securities=1, dropout=0.2, n_layers=2, T=10):
        """
        Constructor of the LSTM based NN for time series prediction

        :param hidden_size: int, size of the first hidden layer
        :param num_securities: int, number of stocks being predicted at the same time
        :param dropout: float, dropout value
        :param n_layers: int, number of layers
        :param T:
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(
            input_size=num_securities,
            hidden_size=self.hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=False
        )

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc1.weight.data.normal_()
        self.fc3 = nn.Linear(self.hidden_size, 10)
        self.fc2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.T = T

    def forward(self, x):
        """
        forward function for LSTM net.
        :param x: Pytorch Variable, T x batch_size x n_stocks
        :return:
        """
        # breakpoint()
        batch_size = x.size()[0]
        # breakpoint()
        seq_length = x.size()[2]
        # breakpoint()

        x = x.view(seq_length, batch_size, -1)  # just to be sure of the dimensions
        # breakpoint()

        # Initial cell states
        h0 = Variable(torch.zeros(self.rnn.num_layers, batch_size, self.hidden_size))
        # breakpoint()

        c0 = Variable(torch.zeros(self.rnn.num_layers, batch_size, self.hidden_size))

        # breakpoint()
        outputs, (ht, ct) = self.rnn(x, (h0, c0))
        # breakpoint()
        out = outputs[-1]  # last prediction
        # breakpoint()
        out = self.fc1(out)
        # breakpoint()
        out = self.fc3(out)
        # breakpoint()
        out = self.relu(out)
        # breakpoint()
        out = self.fc2(out)
        # breakpoint()
        return out
