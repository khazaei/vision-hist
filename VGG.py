"""
https://arxiv.org/pdf/1409.1556.pdf
"""

import torch
import torch.nn as nn

LEARNING_RATE = 0.01
LEARNING_RATE_DECAY_FACTOR = 0.1
LEARNING_RATE_DECAY_STEP_SIZE = 30  # in paper they decay it 3 times
NUM_EPOCHS = 90  # should be 74?
BATCH_SIZE = 256
IMAGE_DIM = 224
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005


# doesn't seem VGG normalizes the data, can remove that??


class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # layer 1
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # layer 2
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # layer 3
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # layer 4
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # layer 5
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # layer 6
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # layer 7
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # layer 8
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # layer 9
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # layer 10
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # layer 11
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # layer 12
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # layer 13
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=7 * 7 * 512, out_features=4096),  # layer 14
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),  # layer 15
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),  # layer 16
        )

        self.init_bias()
        self.optim = torch.optim.AdamW(params=self.parameters(), lr=LEARNING_RATE)
        self.num_epochs = NUM_EPOCHS
        self.batch_size = BATCH_SIZE
        self.in_dim = IMAGE_DIM

    def init_bias(self):
        # In the paper pretraining is done on a smaller depth network to get initial weights. They mentioned they
        # later used initialization using the one proposed in "Understanding the difficulty of training deep
        # feedforward neural networks - Glorot, X. & Bengio, Y. (2010)."
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        y1 = self.conv(x)
        # take the first dimension (batch) and flatten the last 3 layers for the fully connected layers
        y2 = y1.reshape(y1.size(0), -1)
        y3 = self.fc(y2)
        return y3

    def optimizer(self):
        return self.optim

    def lr_scheduler(self):
        return torch.optim.lr_scheduler.StepLR(self.optim, step_size=LEARNING_RATE_DECAY_STEP_SIZE,
                                               gamma=LEARNING_RATE_DECAY_FACTOR)
