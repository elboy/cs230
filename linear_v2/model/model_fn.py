"""Define the model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseballFCN(nn.Module):
    def __init__(self, n_features):
        super(BaseballFCN, self).__init__()
        self.fc1 = nn.Linear(n_features, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 1)

        self.bn1 = nn.BatchNorm1d(200)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        #x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x


class BaseballRNN(nn.Module):
    def __init__(self, n_features):
        super(BaseballFCN, self).__init__()
        self.fc1 = nn.Linear(n_features, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x