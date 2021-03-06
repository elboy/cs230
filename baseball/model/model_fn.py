"""Define the model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseballFCN(nn.Module):
    def __init__(self, n_features, hidden_size):
        super(BaseballFCN, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, 1)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.2, training=self.training)
        #x = F.relu(self.bn3(self.fc3(x)))
        #x = F.dropout(x, p=0.2, training=self.training)
        #x = F.relu(self.bn4(self.fc4(x)))
        #x = F.dropout(x, p=0.2, training=self.training)
        #x = F.relu(self.bn5(self.fc5(x)))
        #x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc6(x))
        return x


class BaseballRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(BaseballRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #  If True, then the input and output tensors are provided as (batch, seq, feature)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.device = device
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # input must be (batch, seq, features)
        output, hn = self.rnn(x, h0)
        pred = self.fc(output[:, -1, :])
        return pred





        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) 
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        #packed = pack_padded_sequence(x.unsqueeze(2), lengths, batch_first=True)
        
        # Forward propagate LSTM
        #packed_hidden, _ = self.lstm(packed, (h0, c0))
        #hidden, _ = pad_packed_sequence(packed_hidden, batch_first=True)
        # Decode the hidden state of the last time step
        #out = self.fc(hidden[np.arange(x.size(0)), lengths, :])
        #return out