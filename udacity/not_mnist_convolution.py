# -*- coding:utf8 -*-
import tensorflow as tf
from six.moves import cPickle as pickle
import numpy as np

pickle_file="/usr/bigdata/data/notMNIST_data/notMNIST.pickle"

with open(pickle_file, "rb") as f:
    save=pickle.load(f)
    train_dataset=save["train_dataset"]
    train_labels=save["train_labels"]
    valid_dataset=save["valid_dataset"]
    valid_labels=save["valid_labels"]
    test_dataset=save["test_dataset"]
    test_labels=save["test_labels"]
    del save
    print("train set", train_dataset.shape, train_labels.shape)
    print("valid set", valid_dataset.shape, valid_labels.shape)
    print("test set", test_dataset.shape, test_labels.shape)


image_size=28
num_labels=10
num_channels=1

def reformat(dataset, labels):
    dataset=dataset.reshape((-1, image_size,image_size, num_channels)).astype(np.float32)
    labels=(np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print("train set", train_dataset.shape, train_labels.shape)
print("valid set", valid_dataset.shape, valid_labels.shape)
print("test set", test_dataset.shape, test_labels.shape)

batch_size=16
patch_size=5
depth=16
num_hidden=64

graph=tf.Graph()
with graph.as_default():
    tf_train_dataset=tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels=tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset=tf.constant(valid_dataset)
    tf_test_dataset=tf.constant(test_dataset)


    layer1_weights=tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases=tf.Variable(tf.zeros([depth]))

    layer2_weights=tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer2_biases=tf.Variable(tf.constant(1.0, shape=[depth])))
 
    layer3_weights=tf.Variable(tf.truncated_normal([image_size//4 * image_size//4 * image_size * depth, num_hidden], stddev=0.1))
    layer3_biases=tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    layer4_weights=tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer4_biases=tf.Variable(tf.constant(1.0, shape=[num_labels]))



    def model(data):
        conv=tf.nn.conv2d(data, layer1_weights, [1,2,2,1], padding="SAME")
        hidden=tf.nn.relu(conv + layer1_biases)

        conv=tf.nn.conv2d(hidden, layer2_weights, [1,2,2,1], padding="SAME")
        hidden=tf.nn.relu(conv + layer2_biases)

        shape=hidden.get_shape().as_list()
        reshape=tf.reshape(hidden, [shape[0], shape[1]*shape[2]*shape[3]])
        hidden=tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases


    logits=model(tf_train_dataset)
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    optimizer=tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    train_prediction=tf.nn.softmax(logits)
    valid_prediction=tf.nn.softmax(model(valid_dataset))
    test_prediction=tf.nn.softmax(model(test_dataset))

num_steps=801

def accuracy(predictions, labels):
    return (100*np.sum(np.argmax(predictions,1)==np.argmax(labels,1))/predictions.shape[0])

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    print("Initialized")

    for step in range(num_steps):
        offset=(step*batch_size) % (train_labels.shape[0] - batch_size)
        batch_dataset=train_dataset[offset:(offset+batch_size), :, :, :]
        batch_labels=train_labels[offset:(offset+batch_size), :]
        _, loss_value,predictions = sess.run([optimizer, loss, train_prediction], feed_dict={tf_train_dataset: batch_dataset, tf_train_labels: batch_labels})
        if step%100==0:
            print("Loss at step %d:%f" % (step, loss_value))
            print("Training accuracy: %.1f%%" % accuracy(predictions, train_labels[:train_subset,:]))
            print("Validation accuracy: %.1f%%" % accuracy(sess.run(valid_prediction),valid_labels))

    print("Test accuracy: %.1f%%" % accuracy(sess.run(test_prediction), test_labels))
