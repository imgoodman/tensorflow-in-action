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

def reformat(dataset, labels):
    dataset=dataset.reshape((-1, image_size*image_size)).astype(np.float32)
    labels=(np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print("train set", train_dataset.shape, train_labels.shape)
print("valid set", valid_dataset.shape, valid_labels.shape)
print("test set", test_dataset.shape, test_labels.shape)

train_subset=20000

graph=tf.Graph()
with graph.as_default():
    tf_train_dataset=tf.constant(train_dataset[:train_subset, :])
    tf_train_labels=tf.constant(train_labels[:train_subset])
    tf_valid_dataset=tf.constant(valid_dataset)
    tf_test_dataset=tf.constant(test_dataset)

    weights=tf.Variable(tf.truncated_normal([image_size*image_size, num_labels]))
    biases=tf.Variable(tf.zeros([num_labels]))

    logits=tf.matmul(tf_train_dataset, weights) + biases
    #prediction=tf.nn.softmax(logits)
    #cross_entropy=-tf.reduce_mean(logits * tf.log(prediction))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
    optimizer=tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    train_prediction=tf.nn.softmax(logits)
    valid_prediction=tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction=tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps=801

def accuracy(predictions, labels):
    return (100*np.sum(np.argmax(predictions,1)==np.argmax(labels,1))/predictions.shape[0])

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    print("Initialized")

    for step in range(num_steps):
        _, loss_value,predictions = sess.run([optimizer, loss, train_prediction])
        if step%100==0:
            print("Loss at step %d:%f" % (step, loss_value))
            print("Training accuracy: %.1f%%" % accuracy(predictions, train_labels[:train_subset,:]))
            print("Validation accuracy: %.1f%%" % accuracy(sess.run(valid_prediction),valid_labels))

    print("Test accuracy: %.1f%%" % accuracy(sess.run(test_prediction), test_labels))
