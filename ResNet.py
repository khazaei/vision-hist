"""
https://arxiv.org/pdf/1512.03385.pdf
"""

import torch
import torch.nn as nn

LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
BATCH_SIZE = 256
IMAGE_DIM = 224
LEARNING_RATE_DECAY_FACTOR = 0.1
LEARNING_RATE_DECAY_STEP_SIZE = 1000
WEIGHT_DECAY = 1e-2


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super().__init__()

        print("creating ResNet")
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(ResidualBlock, 64, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(ResidualBlock, 64, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(ResidualBlock, 128, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Sequential(nn.Linear(512, num_classes))

        print(
            "learning rate {}, weight decay {}, batch size {}, learning rate decay {}, learning rate scheduler step {}".format(
                LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, LEARNING_RATE_DECAY_FACTOR, LEARNING_RATE_DECAY_STEP_SIZE))
        self.init_params()
        self.optim = torch.optim.AdamW(params=self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.num_epochs = NUM_EPOCHS
        self.batch_size = BATCH_SIZE
        self.in_dim = IMAGE_DIM

    def _make_layer(self, block_type, in_channel, out_channel, num_blocks, stride=1):
        downsample = None
        if stride != 1 or in_channel != out_channel:
            downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel),
            )
        layers = []
        layers.append(block_type(in_channel, out_channel, stride, downsample))
        for i in range(1, num_blocks):
            layers.append(block_type(out_channel, out_channel))

        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def optimizer(self):
        return self.optim

    def lr_scheduler(self):
        return torch.optim.lr_scheduler.StepLR(self.optim, step_size=LEARNING_RATE_DECAY_STEP_SIZE,
                                               gamma=LEARNING_RATE_DECAY_FACTOR)


def get_ResNet34(num_classes):
    layers = [3, 4, 6, 3]
    return ResNet(layers, num_classes)
