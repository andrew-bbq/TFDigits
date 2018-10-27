import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

#Dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

learning_rate = .1
num_epochs = 501
batch_size = 128
display_epoch = 50

input_size = 784
num_hidden_1 = 256
num_hidden_2 = 256
num_classes = 10

X = tf.placeholder(tf.float32, [None, input_size])
Y = tf.placeholder(tf.float32, [None, input_size])

#randomize weights
weights = {
    'weight1': tf.Variable(tf.random_normal([input_size, num_hidden_1], stddev = 0.1)),
    'weight2': tf.Variable(tf.random_normal([input_size, num_hidden_2], stddev = 0.1)),
    'output': tf.Variable(tf.random_normal([input_size, num_classes], stddev = 0.1))
}

#randomize biases
biases = {
    'bias1': tf.Variable(tf.constant(0.1, shape=[num_hidden_1])),
    'bias2': tf.Variable(tf.constant(0.1, shape=[num_hidden_2])),
    'output': tf.Variable(tf.constant(0.1, shape=[num_classes]))
}

def ass_network():
    #dot product X & weights for layer 1, add bias for layer 1
    layer_1 = tf.add(tf.matmul(X, weights['weight1']), biases['bias1'])
    #dot product layer_1 and weights for layer 2, add biases for layer 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['weight2']), biases['bias2'])
    #return dot product layer_2 and weights for output, with biases for output added
    return tf.add(tf.matmul(layer_2, weights['output']), biases['output'])

#saves function for future use
prediction = ass_network()
#reduce_mean takes average, softmax_cross_entropy_with_logits compares actual (labels) to prediction (logits)
# compare values of vector actual ex: ([0,0,1,0,...] to less robust guess ex: [0.1, 0.2, 0.8, 0.1,...])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=prediction))
