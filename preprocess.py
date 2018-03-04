import os
import numpy as np
import cv2


def crop(image, height, width, mode='center'):
    _height, _width, _ = image.shape
    if mode == 'center':
        h = int((_height - height) / 2)
        w = int((_width - width) / 2)
    elif mode == 'random':
        h = np.random.randint(0, _height - height - 1)
        w = np.random.randint(0, _width - width - 1)
    return image[h:(h+height), w:(w+width), :]

with open('./data/status.txt', 'r') as f:
    labels = [x.strip() for x in f.readlines()]

    for i,x in enumerate(labels):
        if x == '1':
            labels[i - 1] = '1'

if not os.path.exists('./data/positive/'):
    os.makedirs('./data/positive/')
if not os.path.exists('./data/negative/'):
    os.makedirs('./data/negative/')

images = sorted(os.listdir('./data/original'))
negative = []
positive = []


for i, image in enumerate(images):
    if labels[i] == '1':
        negative.append(image)
    else:
        positive.append(image)

for pos in positive:

    img = cv2.imread('./data/original/' + pos)

    for i in range(10):
        _img = crop(img, 224, 224, mode='random')
        cv2.imwrite('./data/positive/' + pos[:-4] + str(i) + '.jpg', _img)

for neg in negative:

    img = cv2.imread('./data/original/' + neg)

    for i in range(10):
        _img = crop(img, 224, 224, mode='random')
        cv2.imwrite('./data/negative/' + neg[:-4] + str(i) + '.jpg', _img)
