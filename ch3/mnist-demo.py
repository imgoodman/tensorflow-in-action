from tensorflow.examples.tutorials.mnist import input_data

data_path="/usr/bigdata/data/mnist_data/"

mnist=input_data.read_data_sets(data_path, one_hot=True)
