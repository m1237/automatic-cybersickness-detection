import torch
import torch.nn as nn
from torch.autograd import Variable


class CSClassifierCNNLSTM(nn.Module):
    """Class for the CNN-LSTM model for classifying cybersickness.
        - 2 Conv1D layers (64 filters, kernel size = 4)
        - Max pooling: pool size = 2, stride = 2
        - flattening output
        - 1 LSTM layer
        - 2 fully connected layers (1st dense: ReLU activiation, 2nd Softmax)
        - 120 timesteps at once, divided into subsequences of 30 timesteps
        - batch size: 512
        - 13 features
        - 50 epochs training
        - loss function: CCE (categorical cross-entropy loss)
    """

    def __init__(self, num_classes, feature_size, input_size, hidden_size1, hidden_size2, dropout_lstm, dropout_p):
        """
        Args:
            num_classes (int): The number of classes
            feature_size (int): The number of features for the CNN
            input_size (int): The size of the input data for the RNN.
            hidden_size1 (int): The size of the hidden state of the RNN.
            hidden_size2 (int): The size of the hidden state of the dense NN.
            dropout_lstm (float): The dropout probability for the LSTM layer
            dropout_p (float): Dropout probability for the dropout layer
        """
        super(CSClassifierCNNLSTM, self).__init__()
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.dropout_lstm = dropout_lstm
        self.dropout_p = dropout_p

        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=hidden_size1, kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=hidden_size1, out_channels=hidden_size1, kernel_size=4)
        self.pool = nn.MaxPool1d(2, 2)
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size1, batch_first=True,
                            dropout=dropout_lstm)
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
        # CNN
        x = x.permute((0, 2, 1))
        cnn_out1 = self.conv1(x)
        cnn_out1 = self.relu(cnn_out1)
        cnn_out2 = self.conv2(cnn_out1)
        cnn_out2 = self.relu(cnn_out2)
        cnn_out_pooled = self.pool(cnn_out2)
        #  cnn_out_flat = self.flatten(cnn_out_pooled)

        # LSTM
        h_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size1))  # hidden state
        c_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size1))  # internal state
        # Propagate input through LSTM
        # output, (hn, cn) = self.lstm(cnn_out_flat.unsqueeze(1), (h_0, c_0))  # lstm with input, hidden, and internal state
        output, (hn, cn) = self.lstm(cnn_out_pooled.permute(0, 2, 1), (h_0, c_0))
        # hn = hn.view(-1, self.hidden_size1)  # reshaping the data for Dense layer next
        hn = hn.squeeze(0)  # reshaping the data for Dense layer next
        out = self.dropout(hn)

        # Dense
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        # out = self.softmax(out)  # softmax
        return out





