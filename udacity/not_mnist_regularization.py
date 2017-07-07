# -*- coding:utf8 -*-
import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle

pickle_file="/usr/bigdata/data/notMNIST_data/notMNIST.pickle"
train_subset=30000
with open(pickle_file, "rb") as f:
    save=pickle.load(f)
    train_dataset=save["train_dataset"]
    train_labels=save["train_labels"]
    valid_dataset=save["valid_dataset"]
    valid_labels=save["valid_labels"]
    test_dataset=save["test_dataset"]
    test_labels=save["test_labels"]
    del save
    print("Training set",train_dataset.shape, train_labels.shape)
    print("Validation set",valid_dataset.shape, valid_labels.shape)
    print("Test set",test_dataset.shape, test_labels.shape)

image_size=28
num_labels=10

def reformat(dataset, labels):
    dataset=dataset.reshape((-1, image_size*image_size)).astype(np.float32)
    labels=(np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset,labels


train_dataset, train_labels=reformat(train_dataset, train_labels)
train_dataset=train_dataset[:train_subset, :]
train_labels=train_labels[:train_subset, :]
valid_dataset, valid_labels=reformat(valid_dataset, valid_labels)
test_dataset, test_labels=reformat(test_dataset, test_labels)
print("reshaped")
print("Training set", train_dataset.shape, train_labels.shape)
print("Validation set", valid_dataset.shape, valid_labels.shape)
print("Test set", test_dataset.shape, test_labels.shape)



def accuracy(predicitons, labels):
    return (100*np.sum(np.argmax(predictions,1)==np.argmax(labels,1))/predictions.shape[0])


def tf_accuracy(predictions, labels):
    acc=tf.equal(tf.argmax(predictions,1), tf.argmax(labels,1))
    acc=tf.cast(acc, tf.float32)
    return tf.reduce_sum(acc)*100/predictions.shape[0]


batch_size=128
num_steps=200
hidden_nodes=1024
x=tf.placeholder(tf.float32, shape=(batch_size, image_size*image_size),name="x")
y_=tf.placeholder(tf.float32, shape=(batch_size, num_labels),name="y")

weights1=tf.Variable(tf.truncated_normal((image_size*image_size,hidden_nodes)))
biases1=tf.Variable(tf.zeros((hidden_nodes)))

weights2=tf.Variable(tf.truncated_normal((hidden_nodes, num_labels)))
biases2=tf.Variable(tf.zeros((num_labels)))

fc1=tf.matmul(x, weights1)+biases1
fc1=tf.nn.relu(fc1)
fc1=tf.nn.dropout(fc1, keep_prob=0.5)

fc2=tf.matmul(fc1, weights2)+biases2
y=tf.nn.softmax(fc2)

cross_entropy=-tf.reduce_mean(y_ * tf.log(y))
loss=cross_entropy+0.01*(tf.nn.l2_loss(weights1) +tf.nn.l2_loss(weights2))

train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    start_t=0
    for step in range(num_steps):
        xs,ys=train_dataset[start_t:(start_t + batch_size),:],train_labels[start_t:(start_t + batch_size),:]
        start_t=start_t + batch_size
        print("xs, ys", xs.shape,ys.shape)
        sess.run(train_step, feed_dict={x:xs, y_:ys})
        print("Loss of step %d is %f" % (step, sess.run(loss, feed_dict={x:xs,y_:ys})))
