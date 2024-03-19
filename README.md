# vision-hist

A stroll down memory lane: In this series, I'm going to explore the progression of vision neural network architectures
through time. I've selected a few architectures that were groundbreaking for their era. I'll be dissecting their papers,
highlighting the novelty introduced in each architecture, and providing some intuition behind each change. The
architectures we'll be exploring are:

* AlexNet
* VGG
* ResNet
* MobileNet
* Vision Transformers

A side note on the training data: All the papers used the ImageNet dataset for training and validation I use
Imagenette as the dataset to evaluate the different architectures. Imagenette is a subset of 10 easily classified
classes from ImageNet (tench, English springer, cassette player, chainsaw, church, French horn, garbage truck, gas pump,
golf ball, parachute). It's smaller, making it easier to store, and it takes less time to train.

## AlexNet

The first architecture we will explore is AlexNet. This was the architecture introduced in 2012, which made waves when
it achieved a top 5 error rate of 15.3%, 10% lower than its runner-up in the ImageNet competitions. Let's go over some
key contributions of this paper.

### Architecture

The network consists of 5 convolutional layers followed by 3 fully connected layers. For the full details of the
parameters, check out the code, but here is a condensed view of what it looks like:

(Conv->ReLU->LocalNorm->MaxPool) x 2 -> (Conv->ReLU) x 3 -> MaxPool -> (Linear->ReLU) x 3.

### ReLU Activation Function.

Traditionally, neural networks used sigmoid or tanh activation functions. AlexNet uses ReLU as its activation function.
The figure below shows the 3 different activation functions and their derivatives:

![](/Users/hamid/PycharmProjects/vision-hist/assets/act-func.png)

![](/Users/hamid/PycharmProjects/vision-hist/assets/grad-act-func.png)

As you can see, the gradients for ReLU do not saturate for high values when compared to tanh and sigmoid. This helps
with the vanishing gradient problem experienced during backpropagation when training networks. This effect is more
prominent in deeper networks; hence ReLU allows training deep CNNs.

Another feature of ReLU is its computational simplicity compared to tanh and sigmoid. It's a `max(0, x)` operation, and
the gradient is a thresholding operation.

These characteristics lend to faster training of deeper neural networks, leading to better performance.

### GPUs

AlexNet was one of the first networks to use multiple GPUs for training. The neurons in each layer were split across two
GPUs. The GPUs only communicate at certain layers. This enabled larger datasets and faster training times.

### Generalization

Here are some of the techniques they used for generalization:

1 - Data augmentation: They augmented the dataset by generating new images through horizontal reflection, image
translation, and altering the intensity of RGB values.

2 - Drop out regularization: They deployed dropout regularization by randomly zeroing out 50% of the neurons in the
fully connected network
during training.

3 - Local response normalization: Each pixel (x,y location) for each layer is normalized by the sum of all kernel values
at that pixel location within that layer.

### Optimizer

The optimizer they used was SGD with momentum and weight decay. SGD uses a subset (batch) of the dataset to compute the
gradient. This makes the gradient computation a bit noisy compared to using the whole dataset. This helps with
generalization by exploring more of the cost function and helps escape local minima and saddle points. Momentum uses
infinite smoothing or IIR filtering to make the trajectory and the gradient descent algorithm smoother. Weight decay is
another way of saying L2 regularization. L2 regularization helps with overfitting by keeping the weights small, hence
preventing any one path from dominating the prediction.

### Implementation and Results.

These are the hyperparameters used for training:

```
LEARNING_RATE = 0.0001
NUM_EPOCHS = 90
BATCH_SIZE = 64
IMAGE_DIM = 227
LEARNING_RATE_DECAY_FACTOR = 0.1
LEARNING_RATE_DECAY_STEP_SIZE = 30  # in paper they decay it 3 times
WEIGHT_DECAY = 0.1
```

The SGD optimizer mentioned above didn't train. I used an AdamW optimizer which scales the gradient by the RMS value
before the update. Using this optimizer and the hyperparameters, I was able to achieve a 99.9% accuracy on the training
data and a 65% accuracy on the test data.

```
Epoch: 90 	Step: 13270 	Loss: 0.0057 	Acc: 100.0 %
Epoch: 90 	Step: 13280 	Loss: 0.1076 	Acc: 98.4375 %
Epoch: 90 	Step: 13290 	Loss: 0.0149 	Acc: 98.4375 %
Epoch: 90 	Step: 13300 	Loss: 0.0325 	Acc: 98.4375 %
Epoch: 90 	Step: 13310 	Loss: 0.0199 	Acc: 98.4375 %
Epoch: 90 	Step: 13320 	Loss: 0.0065 	Acc: 99.99999237060547 %
Accuracy of the network on the 10000 test images: 65.01910400390625 %
training complete
testing on cuda
Accuracy of the network on the 10000 test images: 64.40763854980469 %
```

let me know if you can achieve better performance by changing the hyperparameters.

## VGG

VGG was introduced in 2014, and its main contribution was increasing the number of layers. There was a theory that
deeper networks were more performant; however, deep networks were difficult to train. Better AIML techniques and GPU
support provided the opportunity to train deeper networks. VGG introduced different configurations, including a 16-layer
network and a 19-layer network.

### Network architecture:

VGG16 network consists of 13 convolutional layers followed by 3 fully connected layers. For the full details of the
parameters, check out the code, but here is a condensed view of what it looks like:

(Conv->ReLU->LocalNorm->MaxPool) x 2 -> (Conv->ReLU) x 3 -> MaxPool -> (Linear->ReLU) x 3

### GPUs

AlexNet split the neurons across the GPUs. This led to some inefficiencies as certain neurons could only communicate
with neurons on the same GPU. In the VGG paper, they used multiple GPUs for data parallelism. They split each batch of
training data across GPUs. Batch gradients on each GPU are computed and averaged to obtain gradients of the full batch.

### Weight Initialization

To initialize the network, the authors pretrained a shallower network, and then used these weights to initialize the
deeper networks. This helped training convergence. Later on, the authors mentioned that Glorot initialization, without
pretraining, resulted in the same performance. Glorot initialization takes into account the fan-in and scales the
weights, which reduces saturation during backpropagation and forward pass.

### Small filters

The receptive field of the CNN was reduced by employing smaller convolutional filters. Most of the filters in the
network are 3x3. The intuition behind smaller filters in a deeper network was that the pace at which the feature
dimensionality was reduced was done at a slower pace. This allowed for the network to learn a rich set of features for
the image classification task.

### Implementation and Results.

The hyperparameters for training are as follows:

```
LEARNING_RATE = 0.0001
NUM_EPOCHS = 90
BATCH_SIZE = 128
IMAGE_DIM = 224
LEARNING_RATE_DECAY_FACTOR = 0.1
LEARNING_RATE_DECAY_STEP_SIZE = 30
WEIGHT_DECAY = 0.1
```

I was able to get the training loss very low, but it didn't generalize well for the test data. Achieving only 68.5%
accuracy on the test data. Contact me if you're able to achieve better results.

```
Epoch: 90 	Step: 6590 	Loss: 0.0001 	Acc: 100.0 %
Epoch: 90 	Step: 6600 	Loss: 0.0006 	Acc: 100.0 %
Epoch: 90 	Step: 6610 	Loss: 0.0006 	Acc: 100.0 %
Epoch: 90 	Step: 6620 	Loss: 0.0013 	Acc: 100.0 %
Epoch: 90 	Step: 6630 	Loss: 0.0010 	Acc: 100.0 %
Epoch: 90 	Step: 6640 	Loss: 0.0001 	Acc: 100.0 %
Epoch: 90 	Step: 6650 	Loss: 0.0000 	Acc: 100.0 %
Epoch: 90 	Step: 6660 	Loss: 0.0001 	Acc: 100.0 %
Accuracy of the network on the 10000 test images: 68.73885345458984 %
training complete
testing on cuda
Accuracy of the network on the 10000 test images: 68.53502655029297 %
```


