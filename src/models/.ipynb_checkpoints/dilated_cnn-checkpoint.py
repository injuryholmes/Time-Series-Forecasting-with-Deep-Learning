from torch import nn

class DilatedNet(nn.Module):
    def __init__(self, num_securities=5, hidden_size=64, dilation=2, T=10):
        """
        :param num_securities: int, number of stocks
        :param hidden_size: int, size of hidden layers
        :param dilation: int, dilation value
        :param T: int, number of look back points
        """
        super(DilatedNet, self).__init__()
        self.dilation = dilation
        self.hidden_size = hidden_size
        # First Layer
        # Input
        self.dilated_conv1 = nn.Conv1d(num_securities, hidden_size, kernel_size=2, dilation=self.dilation)
        self.relu1 = nn.ReLU()

        # Layer 2
        self.dilated_conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=self.dilation)
        self.relu2 = nn.ReLU()

        # Layer 3
        self.dilated_conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=self.dilation)
        self.relu3 = nn.ReLU()

        # Layer 4
        self.dilated_conv4 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, dilation=self.dilation)
        self.relu4 = nn.ReLU()

        # Output layer
        self.conv_final = nn.Conv1d(hidden_size, num_securities, kernel_size=1)

        self.T = T

    def forward(self, x):
        """

        :param x: Pytorch Variable, batch_size x n_stocks x T
        :return:
        """

        # First layer
        out = self.dilated_conv1(x)
        out = self.relu1(out)

        # Layer 2:
        out = self.dilated_conv2(out)
        out = self.relu2(out)

        # Layer 3:
        out = self.dilated_conv3(out)
        out = self.relu3(out)

        # Layer 4:
        out = self.dilated_conv4(out)
        out = self.relu4(out)

        # Final layer
        out = self.conv_final(out)
        out = out[:, :, -1]

        return out


class DilatedNet2D(nn.Module):
    def __init__(self, hidden_size=64, dilation=1, T=10):
        """

        :param hidden_size: int, size of hidden layers
        :param dilation: int, dilation value in the time dimension (1 for the other dimension, aka between the stocks)
        :param T: int, number of look back points
        """
        super(DilatedNet2D, self).__init__()
        self.dilation = dilation
        self.hidden_size = hidden_size
        # First Layer
        # Input
        self.dilated_conv1 = nn.Conv2d(1, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu1 = nn.ReLU()

        # Layer 2
        self.dilated_conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu2 = nn.ReLU()

        # Layer 3
        self.dilated_conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu3 = nn.ReLU()

        # Layer 4
        self.dilated_conv4 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu4 = nn.ReLU()

        # Output layer
        self.conv_final = nn.Conv2d(hidden_size, 1, kernel_size=(1, 2))

        self.T = T

    def forward(self, x):
        """

        :param x: Pytorch Variable, batch_size x 1 x n_stocks x T
        :return:
        """

        # First layer
        out = self.dilated_conv1(x)
        out = self.relu1(out)

        # Layer 2:
        out = self.dilated_conv2(out)
        out = self.relu2(out)

        # Layer 3:
        out = self.dilated_conv3(out)
        out = self.relu3(out)

        # Layer 4:
        out = self.dilated_conv4(out)
        out = self.relu4(out)

        # Final layer
        out = self.conv_final(out)
        out = out[:, :, :, -1]

        return out


class DilatedNet2DMultistep(nn.Module):
    def __init__(self, num_securities=5, n_in=20, n_out=3, hidden_size=64, dilation=1, T=10):
        """

        :param num_securities:
        :param n_in: number of time points in the input
        :param n_out: number of time points in the output
        :param hidden_size: int
        :param dilation: int
        :param T: int, length of lookback
        """
        super(DilatedNet2DMultistep, self).__init__()
        self.n_out = n_out
        self.n_in = n_in
        self.dilation = dilation
        self.hidden_size = hidden_size
        # First Layer
        # Input
        self.dilated_conv1 = nn.Conv2d(1, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))  # dilation in
                                                                                                    # the time dimension
        self.relu1 = nn.ReLU()

        # Layer 2
        self.dilated_conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu2 = nn.ReLU()

        # Layer 3
        self.dilated_conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu3 = nn.ReLU()

        # Layer 4
        self.dilated_conv4 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 2), dilation=(1, self.dilation))
        self.relu4 = nn.ReLU()

        # Output layer
        self.conv_final = nn.Conv2d(hidden_size, 1, kernel_size=(1, 2))

        self.T = T

    def forward(self, x):
        """

        :param x: Pytorch Variable, batch_size x 1 x T x n_stocks
        :return:
        """

        # First layer
        out = self.dilated_conv1(x)
        out = self.relu1(out)

        # Layer 2:
        out = self.dilated_conv2(out)
        out = self.relu2(out)

        # Layer 3:
        out = self.dilated_conv3(out)
        out = self.relu3(out)

        # Layer 4:
        out = self.dilated_conv4(out)
        out = self.relu4(out)

        # Final layer
        out = self.conv_final(out)
        out = out[:, :, :, -self.n_out:]

        return out

