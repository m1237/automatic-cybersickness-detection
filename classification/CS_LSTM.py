import torch
import torch.nn as nn
from torch.autograd import Variable


class CSClassifierLSTM(nn.Module):
    """Class for the LSTM model for classifying cybersickness.
        - 4 layer LSTM
        - 120 timesteps at once (?)
        - batch size: 512
        - 13 features
        - 50 epochs training
        - 1st layer: LSTM, 0.5 dropout
        - 2nd layer: 0.4 dropout
        - 3rd layer: dense, ReLU as activation
        - 4th layer: dense, softmax as activation, 3 outputs (because they classify 3 stages of CS)
        - loss function: CCE (categorical cross-entropy loss)
    """

    def __init__(self, num_classes, input_size, hidden_size1, hidden_size2, dropout_lstm, dropout_p):
        """
        Args:
            num_classes (int): The number of classes
            input_size (int): The size of the input data for the RNN.
            hidden_size1 (int): The size of the hidden state of the RNN.
            hidden_size2 (int): The size of the hidden state of the dense NN.
            dropout_lstm (float): The dropout probability for the LSTM layer
            dropout_p (float): Dropout probability for the dropout layer
        """
        super(CSClassifierLSTM, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.dropout_lstm = dropout_lstm
        self.dropout_p = dropout_p

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size1, batch_first=True,
                                          dropout=dropout_lstm)  # LSTM
        self.dropout = nn.Dropout(dropout_p)
        self.fc_1 = nn.Linear(hidden_size1, hidden_size2)  # fully connected 1
        self.relu = nn.ReLU()  # ReLU as activation function
        self.fc = nn.Linear(hidden_size2, num_classes)  # fully connected last layer
        self.softmax = nn.Softmax(dim=1)  # softmax as activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (tensor): The feature tensor (n_rows, n_timesteps, n_features)
        Returns:
            out (tensor): The output after forward pass (2*n_rows, n_classes)
        """
        # LSTM
        h_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size1))  # hidden state
        c_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size1))  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        # hn = hn.view(-1, self.hidden_size1)  # reshaping the data for Dense layer next
        hn = hn.squeeze(0)  # reshaping the data for Dense layer next
        out = self.dropout(hn)

        # Dense
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        # out = self.softmax(out)  # softmax
        return out





