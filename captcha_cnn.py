import torch.nn as nn


class CaptchaCNN(nn.Module):
    def __init__(self):
        super(CaptchaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 14 * 4, 1024)
        self.dropout = nn.Dropout(0.5)  # 添加Dropout层
        self.fc2 = nn.Linear(1024, 4 * 10)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2, 2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2, 2)(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.MaxPool2d(2, 2)(x)
        x = x.view(-1, 128 * 14 * 4)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)  # 应用Dropout
        x = self.fc2(x)
        x = x.view(-1, 4, 10)
        return x

