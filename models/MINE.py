import torch
import torch.nn as nn

class MINE(nn.Module):
    def __init__(self, input_size=256, hidden_size=128):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        # 连接两个输入的特征
        xy = torch.cat((x, y), dim=1)
        h = self.relu(self.fc1(xy))
        h = self.relu(self.fc2(h))
        return self.fc3(h)
