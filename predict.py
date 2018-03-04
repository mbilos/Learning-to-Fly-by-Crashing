import os
import tensorflow as tf
import numpy as np
import cv2
import time

def read_image(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image, (224, 224))
    image = np.divide(image, 255.0)
    image = np.expand_dims(image, axis=0)

    return image

images = sorted(os.listdir('./data/frames'))
predictions = []

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./models/model.ckpt.meta')
    saver.restore(sess, './models/model.ckpt')

    start = time.time()

    for img in images:
        img = read_image('./data/frames/' + img)
        predictions.append(str(sess.run('pred/pred:0', { 'x:0': img })[0]))

    end = time.time()
    print("Time:", end - start, "\tper image:", (end - start) / len(images))

with open('./data/predictions.txt', 'w') as f:
    f.writelines('\n'.join(predictions))


for name, pred in zip(images, predictions):
    color = (0, 255, 0) if pred == 0 else (0, 0, 255)
    img = cv2.imread('./data/frames/' + name)
    cv2.circle(img, (600,320), 20, color, -1)
    cv2.imwrite('./data/processed_frames/' + name, img)
