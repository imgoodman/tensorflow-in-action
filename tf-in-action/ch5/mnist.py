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
    #初始化滑动平均类
    ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    #在所有代表神经网络参数的变量上使用滑动平均
    variables_averages_op=ema.apply(tf.trainable_variables())
    #计算使用了滑动平均之后的前向传播结果
    #滑动平均不会改变变量本身的取值 而是维护一个影子变量来记录其滑动平均值。所以当需要使用这个滑动平均值的时候，需要明确调用average函数
    average_y=inference(x, variable_averages, weights1, biases1, weights2, biases2)
    #第一个参数是神经网络的前向传播结果
    #第二个参数是训练数据的正确答案的数字（所以要用tf.argmax）
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    #计算L2正则化损失函数
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算模型的正则化损失
    regularization=regularizer(weights1)+regularizer(weights2)
    #总得损失
    loss=cross_entropy_mean + regularization
    #设置指数衰减的学习率
    #基础学习率，当前迭代的轮数，过完所有的训练数据需要的迭代次数，学习率衰减速度
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
    #在训练神经网络模型时候，每过一遍数据，既需要通过反向传播来更新网络中的参数，又要更新每一个参数的滑动平均值
    #为了一次完成多个操作
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op=tf.no_op(name="train")
    #tf.argmax的第二个参数“1”表示选取最大值的操作仅仅在第一个维度中进行。也就是说，只在每一行选取最大值对应的下标
    #tf.equal判断两个张量在每一维是否相等
    correct_prediction=tf.equal(tf.argmax(average_y,1), tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        #初始化应该采用tf.global_variables_initializer().run()
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
