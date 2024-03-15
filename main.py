import AlexNet
import train
import pytorch_dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import os

NUM_CLASSES = 10


def main():
    model = AlexNet.AlexNet(num_classes=NUM_CLASSES)
    print('model created')
    print(model)

    transform = transforms.Compose([
        transforms.CenterCrop(model.in_dim),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    download = True
    if os.path.isdir('data'):
        download = False

    train_data = pytorch_dataset.Imagenet10(data_set_path='./data', batch_size=model.batch_size, transform=transform,
                                            train_set=True, download=download)
    print('data loaded')

    train.train(model, train_data)

    test_data = pytorch_dataset.Imagenet10(data_set_path='./data', batch_size=model.batch_size, transform=transform,
                                           train_set=False, download=False)
    train.test(model, test_data)


if __name__ == '__main__':
    main()


def plot(images):
    img = torchvision.utils.make_grid(images)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
