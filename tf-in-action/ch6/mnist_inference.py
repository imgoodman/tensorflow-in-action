# -*- coding:utf8 -*-
import tensorflow as tf

INPUT_NODE=784
OUTPUT_NODE=10

IMAGE_SIZE=28
NUM_CHANNELS=1
NUM_LABELS=10

CONV1_DEEP=32
CONV1_SIZE=5


CONV2_DEEP=64
CONV2_SIZE=5

FC_SIZE=512

"""
输入28*28*1
"""
def inference(input_tensor, train, regularizer):
    with tf.variable_scope("layer1-conv1"):
        """
        卷积层的输入：28*28*1，图片宽度28，高度28，通道数1
        过滤器尺寸：5*5
        过滤器深度：32（卷积核数量）
        全0填充（在外围填充两圈的0，才能保证图像的每个元素都是尺寸5的卷积核的中心）
        卷积层的输出：28*28*32
        步长：1*1
        """
        conv1_weights=tf.get_variable("weight", shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1) )
        conv1_biases=tf.get_variable("bias", shape=[CONV1_DEEP], initializer=tf.constant_initializer(0.1))
        conv1=tf.nn.conv2d(input_tensor, conv1_weights, strides=[1,1,1,1], padding="SAME")
        bias1=tf.nn.bias_add(conv1, conv1_biases)
        relu1=tf.nn.relu(bias1)

    with tf.variable_scope("layer2-pool1"):
        """
        池化层的输入：28*28*32
        过滤器尺寸：2*2
        步长：2*2
        池化层输出：14*14*32
        """
        pool1=tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        """
        卷积层输入：14*14*32
        过滤器尺寸：5*5
        过滤器深度（卷积核数量）：64
        全0填充
        卷积层输出：14*14*64
        步长：1*1
        """
        conv2_weights=tf.get_variable("weights", shape=[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases=tf.get_variable("biases", shape=[CONV2_DEEP], initializer=tf.constant_initializer(0.1))
        conv2=tf.nn.conv2d(relu1, conv2_weights, strides=[1,1,1,1], padding="SAME")
        bias2=tf.nn.bias_add(conv2, conv2_biases)
        relu2=tf.nn.relu(bias2)

    with tf.variable_scope("layer4-pool2"):
        """
        池化层输入：14*14*64
        过滤器尺寸：2*2
        步长：2*2
        池化层输出：7*7*64
        """
        pool2=tf.nn.max_pool(relu2, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    """
    池化层输出：7*7*64
    全连接层的输入格式为向量
    因此，需要将池化层的输出拉直成一个向量
    每一层神经网络的输入输出都是一个batch_size的矩阵
    注意：pool_shape[0]是batch_size
    拉直成为向量，其长度=矩阵长度*矩阵宽度*矩阵深度
    """
    pool_shape=pool2.get_shape().as_list()
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]

    reshaped=tf.reshape(pool2, [pool_shape[0], nodes])


    with tf.variable_scope("layer5-fc1"):
        fc1_weights=tf.get_variable("weights", shape=[nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection("losses", regularizer(fc1_weights))
        fc1_biases=tf.get_variable("biases", shape=[FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1=tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1=tf.nn.dropout(fc1, 0.5)



    with tf.variable_scope("layer6-fc2"):
        fc2_weights=tf.get_variable("weights", shape=[FC_SIZE, NUM_LABELS], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection("losses", regularizer(fc2_weights))
        fc2_biases=tf.get_variable("biases", shape=[NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit=tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit
