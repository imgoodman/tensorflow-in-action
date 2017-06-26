#-*-coding:utf8-*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data_path="/usr/bigdata/data/mnist_data/"

#输入层节点数
INPUT_NODE=784
#输出层节点数
OUTPUT_NODE=10

#隐含层节点数
LAYER1_NODE=500
#一个训练batch中的训练数据个数
BATCH_SIZE=100

#基础学习率
LEARNING_RATE_BASE=0.8
#学习率的衰减率
LEARNING_RATE_DECAY=0.99

#防止过拟合  正则化程度
REGULARIZATION_RATE=0.0001

#总得训练迭代次数
TRAINING_STEPS=30000
#滑动平均衰减率
MOVING_AVERAGE_DECAY=0.99


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class is None:
        layer1=tf.nn.relu(tf.matmul(input_tensor, weights1)+biases1)
        return tf.matmul(layer1, weights2)+biases2
    else:
        layer1=tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    x=tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    y_=tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")
    
    weights1=tf.Variable(tf.truncated_normal(shape=[INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    weights2=tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y=inference(x, None, weights1, biases1, weights2, biases2)

    global_step=tf.Variable(0, trainable=False)

    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    variables_averages_op=variable_averages.apply(tf.trainable_variables())

    average_y=inference(x, variable_averages, weights1, biases1, weights2, biases2)

    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)

    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    regularization=regularizer(weights1)+regularizer(weights2)
    loss=cross_entropy_mean + regularization

    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op=tf.no_op(name="train")

    correct_prediction=tf.equal(tf.argmax(average_y,1), tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed={x:mnist.validation.images, y_:mnist.validation.labels}
        test_feed={x:mnist.test.images, y_:mnist.test.labels}
        for i in range(TRAINING_STEPS):
            if i%100==0:
                validate_accuracy=sess.run(accuracy, feed_dict=validate_feed)
                print("after %d training steps, validation accuracy using average model is %g" % (i, validate_accuracy))
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x:xs, y_:ys})
        test_accuracy=sess.run(accuracy, feed_dict=test_feed)
        print("after %d training steps, test accuracy using average model is %g" % (TRAINING_STEPS, test_accuracy))



def main(argv=None):
    mnist=input_data.read_data_sets(data_path, one_hot=True)
    train(mnist)



if __name__=="__main__":
    tf.app.run()
