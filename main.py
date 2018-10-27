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

#placeholders for input
X = tf.placeholder(tf.float32, [None, input_size])
Y = tf.placeholder(tf.float32, [None, input_size])

# randomize weights
weights = {
    'weight1': tf.Variable(tf.random_normal([input_size, num_hidden_1], stddev = 0.1)),
    'weight2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], stddev = 0.1)),
    'output': tf.Variable(tf.random_normal([num_hidden_2, num_classes], stddev = 0.1))
}

# randomize biases - vector of correct sizes to add to 
biases = {
    'bias1': tf.Variable(tf.constant(0.1, shape=[num_hidden_1])),
    'bias2': tf.Variable(tf.constant(0.1, shape=[num_hidden_2])),
    'output': tf.Variable(tf.constant(0.1, shape=[num_classes]))
}

def ass_network():
    # cross product X & weights for layer 1, add bias for layer 1
    layer_1 = tf.add(tf.matmul(X, weights['weight1']), biases['bias1'])
    # cross product layer_1 and weights for layer 2, add biases for layer 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['weight2']), biases['bias2'])
    # return cross product layer_2 and weights for output, with biases for output added
    return tf.add(tf.matmul(layer_2, weights['output']), biases['output'])

# saves function for future use
prediction = ass_network()
# reduce_mean takes average, softmax_cross_entropy_with_logits compares actual (labels) to prediction (logits)
# compare values of vector actual ex: ([0,0,1,0,...] to less robust guess ex: [0.1, 0.2, 0.8, 0.1,...])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=prediction))

# define training operation, documentation for this makes the math look really hard so I'll trust it does what it's supposed to
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# define correct prediction, if index of highest prediction are the same, return 1 if correct and 0 if not
prediction_correct = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))

# percentage of predictions which are correct
accuracy = tf.reduce_mean(tf.cast(prediction_correct, tf.float32))

#initialize global variables from beginning
init = tf.global_variables_initializer()

#initialize tensorflow session
sess = tf.Session()
sess.run(init)

#for number of epochs
for epoch in range(1, num_epochs):
    #batch_x as images, batch_y as labels
    #gets next $batch_size from mnist database
    batch_x, batch_y = mnist.train.next_batch(batch_size)

    #session run neural network from beginning - train function as parameter- includes loss function as parameter which includes prediction as paramter
    #train is kind of like "final" function
    #fill place holder with data from training set (X: batch_x, Y: batch_y)
    sess.run(train, feed_dict = {X: batch_x, Y: batch_y})
    
    #if first run or multiple of $display_epoch then display training data
    if epoch == 1 or epoch % display_epoch == 0:
        batch_loss, model_accuracy = sess.run([loss, accuracy], feed_dict = {X: batch_x, Y: batch_y})
        print("EPOCH " + str(epoch) + ", BATCH LOSS: " + \
        "{:.4f}".format(batch_loss) + ", TRAINING ACCURACY: "+ \
        "{:.3f}".format(model_accuracy)
        )