# Learning to fly by crashing

Implementation of Ghandi et al. [paper](https://arxiv.org/abs/1704.05588) in tensorflow with data from simulation.

Dependencies:
```
python 3.6
tensorflow 1.4
numpy
cv2
```

Directory tree should look like this:

```
Learning-to-Fly-by-Crashing
|-- models
|-- logs
|-- data
|   |-- status.txt
|   |-- original
|       |-- *.jpg
|-- predict.py
|-- preprocess.py
|-- train.py
```

There are 10 example images and 10 image labels inside status.txt. Zero (0) denotes normal flying and one (1) is a moment of crash. Running preprocess.py will split images based on their label into positive and negative directory. Since images are larger than input of neural network, each one is cropped randomly 10 times which results in 10 times bigger dataset.

Neural network arhitecture is based on [AlexNet](https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf) but instead of object recognition with 1000 classes, it predicts whether the drone will collide with something or not. Running train.py will train network and save weights.