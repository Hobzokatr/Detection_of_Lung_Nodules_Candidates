import torch
import torch.nn as nn
import torch.nn.functional as F

"""The Laplacian of Gaussian"""

class LoG_Blob_Detector(nn.Module):
    def __init__(self, height, width, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv4(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv5(x))
        x = F.sigmoid(self.conv6(x))

        return x

"""Difference of Gaussians"""
class DoG_Blob_Detector(nn.Module):
    def __init__(self, height, width, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = self.up1(x)
        x = F.relu(self.conv4(x))
        x = self.up2(x)
        x = F.relu(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))

        return x

"""The determinant of the Hessian"""
class DoH_blob_detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=(5, 5), stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(in_features=16 * (height // 4) * (width // 4), out_features=150)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(in_features=150, out_features=100)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(in_features=100, out_features=50)
        self.fc4 = nn.Linear(in_features=50, out_features=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * (height // 4) * (width // 4))
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


