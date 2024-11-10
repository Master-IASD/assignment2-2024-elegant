import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(100, 256)
        self.norm1 = nn.BatchNorm1d(self.fc1.out_features)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.norm2 = nn.BatchNorm1d(self.fc2.out_features)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.norm3 = nn.BatchNorm1d(self.fc3.out_features)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.norm1(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.norm2(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.norm3(x)
        return torch.tanh(self.fc4(x))

class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
        self.dropout = nn.Dropout(0.1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.1)
        #x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.1)
        #x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.1)
        #x = self.dropout(x)
        return self.fc4(x)
