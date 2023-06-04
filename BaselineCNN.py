import torch as th
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    Basic CNN model to be used as baseline for artist classification of paintings.
    """

    def __init__(self, num_classes=57):
        """
        Inits the layers of the model with the default sizes from the paper.
        """
        super(BaselineCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv_bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv_bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(6272, 228)
        self.fc_bn1 = nn.BatchNorm1d(228)
        self.fc2 = nn.Linear(228, num_classes)

    def forward(self, x):
        """
        Pushes data through layers and adds the ReLU's in-between each layer.
        """
        x = F.relu(self.conv_bn1(self.maxpool1(self.conv1(x))))
        x = F.relu(self.conv_bn2(self.maxpool2(self.conv2(x))))
        x = th.flatten(x)
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.fc2(x)
        ret = F.softmax(x, dim=1)

        return ret
