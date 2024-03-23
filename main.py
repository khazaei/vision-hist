import AlexNet
import VGG
import ResNet
import train
import pytorch_dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import os
from torchsummary import summary


NUM_CLASSES = 10


def main():
    model = ResNet.ResNet(num_classes=NUM_CLASSES)
    print('model created')
    summary(model, (3, model.in_dim, model.in_dim))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(model.in_dim, model.in_dim), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ])

    download = True
    if os.path.isdir('data'):
        download = False

    train_data = pytorch_dataset.Imagenet10(data_set_path='./data', batch_size=model.batch_size, transform=transform,
                                            train_set=True, download=download)
    print('data loaded')

    transform_test = transforms.Compose([
        transforms.Resize((model.in_dim, model.in_dim)),
        transforms.ToTensor(),
        normalize,
    ])
    test_data = pytorch_dataset.Imagenet10(data_set_path='./data', batch_size=model.batch_size,
                                           transform=transform_test,
                                           train_set=False, download=False)

    train.train(model, train_data, test_data)
    train.test(model, test_data)


if __name__ == '__main__':
    main()


def plot(images):
    img = torchvision.utils.make_grid(images)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
