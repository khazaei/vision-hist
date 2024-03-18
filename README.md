# vision-hist

A stroll down memory lane. Exploring progression of vision neural network architectures through time.

## AlexNet

The first architecture we will explore is AlexNet. This was the architecture introduced in 2012, made waves when it
achieved a top 5 error of 15.3%, 10% lower than its runner-up. Here are some highlights from the paper:

* 5 convolutional layer, 3 fully connected layers.
* ReLU activation. Enabled faster training for deep networks.
* Training on 2 GPUs. The neurons in each layer, split between the GPUs. Enable larger datasets and faster training.
* Local response normalization. Each pixel (x,y location) for each layer, is normalized by the sum of all kernel values
  at that pixel location within that layer. Aided generalization.
* overlapping max pooling.
* (Conv->ReLU->LocalNorm->MaxPool) x 2 -> (Conv->ReLU) x 3 -> MaxPool -> (Linear->ReLU) x 3
* For generalization:
    * Data augmentation: horizontal reflection, image translation. altering intensity of RGB values.
    * Dropout regularization
    * SGD with momentum and weight decay (l2 regulation).

## VGG

VGG was introduced in 2014 and its main contribution was increasing the number of layers. different configurations were
introduced including a 16 layer network and a 19 layer network. In addition, the receptive field was reduced by making
use of smaller convolutional filters. The intuition being that the large depth of the cnn allows one to reduce the
feature vectors at a smaller pace.

Network architecture:

* (Conv->ReLU->LocalNorm->MaxPool) x 2 -> (Conv->ReLU) x 3 -> MaxPool -> (Linear->ReLU) x 3
* 4 GPUs used for data parallelism. split each batch of training across GPUs. Batch gradients are computed and averaged
  to obtain gradients of the full batch.
* Glorot initialization.

# Dataset

I use Imagenette as the data set to evaluate the different architectures. Imagenette is a subset of 10 easily classified
classes from Imagenet (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas
pump, golf ball, parachute). Its smaller, so easier to store, and it takes less time to train.