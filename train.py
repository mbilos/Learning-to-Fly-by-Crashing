import os
import numpy as np
import cv2
import tensorflow as tf
import datetime

slim = tf.contrib.slim
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def read_image(filename):
    img = cv2.imread(filename)
    img = np.divide(img, 255.0)

    return img

def get_data():
    positive = ['./data/positive/' + x for x in os.listdir('./data/positive')]
    negative = ['./data/negative/' + x for x in os.listdir('./data/negative')]

    data = np.array(positive + negative)
    labels = np.array([0] * len(positive) + [1] * len(negative))

    p = np.random.permutation(len(data))
    data = data[p]
    labels = labels[p]
    N = len(data)

    x_train, y_train = data[:int(N * 0.7)], labels[:int(N * 0.7)]
    x_val, y_val = data[int(N * 0.7):int(N * 0.85)], labels[int(N * 0.7):int(N * 0.85)]
    x_test, y_test = data[int(N * 0.85):], labels[int(N * 0.85):]

    return x_train, y_train, x_val, y_val, x_test, y_test

x_train, y_train, x_val, y_val, x_test, y_test = get_data()


BATCH_SIZE = 64
ITERATIONS = 1000
LEARNING_RATE = 0.0001
L2 = 0.00001


def get_batch(data, labels, size):
    indexes = np.random.randint(0, len(data), size)
    data = data[indexes]
    labels = labels[indexes]

    images = []
    for d in data:
        images.append(read_image(d))
    return np.array(images), labels

def print_accuracy(labels, val_pred):
    true_positive = len([x for x,y in zip(labels, val_pred) if x == y and y == 1])
    print('correct positive:', true_positive, 'of', sum(labels == 1), 'positive',
          round(true_positive / sum(labels == 1), 4))
    true_negative = len([x for x,y in zip(labels, val_pred) if x == y and y == 0])
    print('correct negative:', true_negative, 'of', sum(labels == 0), 'negative',
          round(true_negative / sum(labels == 0), 4))

with tf.Graph().as_default():

    x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x')
    y = tf.placeholder(tf.int32, [None], name='y')

    net = slim.conv2d(x, 64, [11, 11], 4, padding='VALID', scope='conv1')   # (?, 54, 54, 64)
    net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')                    # (?, 26, 26, 64)
    net = slim.conv2d(net, 192, [5, 5], scope='conv2')                      # (?, 26, 26, 192)
    net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')                    # (?, 12, 12, 192)
    net = slim.conv2d(net, 384, [3, 3], scope='conv3')                      # (?, 12, 12, 384)
    net = slim.conv2d(net, 384, [3, 3], scope='conv4')                      # (?, 12, 12, 384)
    net = slim.conv2d(net, 256, [3, 3], scope='conv5')                      # (?, 12, 12, 256)
    net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')                    # (?, 5, 5, 256)
    net = slim.conv2d(net, 4096, [5, 5], padding='VALID', scope='fc6')      # (?, 1, 1, 4096)
    net = slim.dropout(net, 0.5, scope='dropout6')                          # (?, 1, 1, 4096)
    net = slim.conv2d(net, 1024, [1, 1], scope='fc7')                       # (?, 1, 1, 1024)
    net = slim.dropout(net, 0.5, scope='dropout7')                          # (?, 1, 1, 1024)
    net = slim.conv2d(net, 2, [1, 1], activation_fn=None, normalizer_fn=None,
        biases_initializer=tf.zeros_initializer(), scope='fc8')             # (?, 1, 1, 2)
    net = tf.reshape(net, [-1, 2])                                          # (?, 2)


    with tf.variable_scope('pred'):
        pred_label = tf.argmax(net, 1, name='pred')
        softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=net)

    with tf.variable_scope('cost'):
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
        loss = tf.reduce_mean(softmax) + lossL2 * L2

    optimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    tf.summary.scalar("cost_gru_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), loss)
    summary_batch = tf.summary.merge_all()

    writer = tf.summary.FileWriter('./logs')
    writer_train = tf.summary.FileWriter('./logs/dmn/plot_train')
    writer_val = tf.summary.FileWriter('./logs/dmn/plot_val')
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)


        for i in range(ITERATIONS):

            data, labels = get_batch(x_train, y_train, BATCH_SIZE)
            feed = { x: data, y: labels }
            _loss, _ = sess.run([loss, optimize], feed)

            if i % 10 == 0:

                summary = sess.run(summary_batch, feed)
                writer_train.add_summary(summary, i)

                data, labels = get_batch(x_val, y_val, 100)
                val_feed = { x: data, y: labels }
                summary_val, val_pred = sess.run([summary_batch, pred_label], val_feed)
                writer_val.add_summary(summary_val, i)

                print("Iteration:", i, "\tof", ITERATIONS)
                print_accuracy(labels, val_pred)


        print('TEST SET')
        _pred = []
        for i in range(len(x_test)):
            img = np.expand_dims(read_image(x_test[i]), axis=0)
            _pred.append(sess.run(pred_label, { x: img }))
        print_accuracy(y_test, _pred)

        saver.save(sess, './models/model.ckpt')
