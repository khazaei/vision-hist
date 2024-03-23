"""
https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
"""

import torch
import torch.nn as nn

LEARNING_RATE = 0.0001
NUM_EPOCHS = 200
BATCH_SIZE = 256
IMAGE_DIM = 227
LEARNING_RATE_DECAY_FACTOR = 0.1
LEARNING_RATE_DECAY_STEP_SIZE = 500  # in paper they decay it 3 times
WEIGHT_DECAY = 0.01

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.conv = nn.Sequential(
            # paper says 224 x 224?
            # in = 227 x 227 x 3
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # out = 96 x 55 x 55,
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # hyper parameters from paper
            nn.MaxPool2d(kernel_size=3, stride=2),  # out = 96 x 27 x 27,
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),  # out = 256 x 27 x 27
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # out = 256 x 13 x 13
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),  # out = 384 x 13 x 13
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),  # out = 384 x 13 x 13
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),  # out = 384 x 13 x 13
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # out = 256 x 6 x 6
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

        print("creating AlexNet")
        print(
            "learning rate {}, weight decay {}, batch size {}, learning rate decay {}, learning rate scheduler step {}".format(
                LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, LEARNING_RATE_DECAY_FACTOR, LEARNING_RATE_DECAY_STEP_SIZE))

        self.init_params()
        self.optim = torch.optim.AdamW(params=self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.num_epochs = NUM_EPOCHS
        self.batch_size = BATCH_SIZE
        self.in_dim = IMAGE_DIM

    def init_params(self):
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

        nn.init.constant_(self.conv[4].bias, 1)
        nn.init.constant_(self.conv[10].bias, 1)
        nn.init.constant_(self.conv[12].bias, 1)

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
