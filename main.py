import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

#Dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

