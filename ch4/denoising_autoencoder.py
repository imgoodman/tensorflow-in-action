#-*-coding:utf8-*-
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import sklearn.preprocessing as prep

data_path="/usr/bigdata/data/mnist_data/"

#参数初始化方法
#根据某一层网络的输入、输出节点数量自动调整最合适的分布
#如果深度学习模型权重初始化太小，信号将在每层间传递时逐渐缩小而难以产生作用
#如果权重初始化太大，信号将在每层间传递时逐渐放大并导致发散和失效
#Xavier就是让权重满足0均值，同时方差为2/(n_in + n_out)，分布可以是均匀分布，也可以是高斯分布
#fan_in输入节点的数量
#fan_out输出节点的数量
def xavier_init(fan_in, fan_out, constant=1):
    low=-constant*np.sqrt(6.0/(fan_in + fan_out))
    high=constant*np.sqrt(6.0/(fan_in + fan_out))
    #tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
    #均匀分布随机数,范围[minval, maxval]
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
    #其它生成随机数tensor
    #tf.random_noral(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
    #正态分布随机数，均值mean，标准差stddev
    #tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
    #截断正态分布随机数，均值mean，标准差stddev，只保留[mean-2*stddev, mean+2*stddev]范围内的随机数

#加性高斯噪声去噪自编码器
class AdditiveGaussianNoiseAutoencoder(object):
    #构造函数
    #n_input输入节点数
    #n_hidde隐含层节点数
    #transfer_function隐含层激活函数，默认softplus
    #optimizer优化器，默认Adam
    #scale高斯噪声系数，默认0.1
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.transfer=transfer_function
        #scale参数是placeholder
        self.scale=tf.placeholder(tf.float32)
        self.training_scale=scale
        #参数初始化
        network_weights=self._initialize_weights()
        self.weights=network_weights
        #开始定义网络结构
        #输入x，（None，n_input]
        self.x=tf.placeholder(tf.float32, [None, n_input])
        #建立隐含层
        #self.x + scale * tf.random_normal((n_input,))，将输入加上噪声
        #注意，这里tf.random_normal传入的shape参数，第二维度没有指明，得到的矩阵是1*n_input（行向量）
        #这样，怎么能够跟x相加呢？有点不明白
        #tf.matmul将加了噪声的输入与隐含层权重w1相乘
        #tf.add将上述相乘结果加上隐含层的偏置b1
        #使用transfer激活函数对结果进行处理
        self.hidden=self.transfer( tf.add( tf.matmul( self.x + scale*tf.random_normal((n_input,)),self.weights['w1']  ), self.weights['b1']  )     )
        #经过隐含层，在输出层进行数据复原、重建操作
        #注意，这里不需要激活函数（为什么这里不需要激活函数呢？）
        self.reconstruction=tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        #定义自编码器的损失函数
        #使用平方误差作为cost
        #tf.subtract计算输出和输入的差
        #tf.pow求差的平方
        #tf.reduce_sum求和
        self.cost=0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0 ) )
        #优化器，对cost进行优化
        self.optimizer=optimizer.minimize(self.cost)
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        #初始化自编码器的全部模型参数
        self.sess.run(init)
    
    def _initialize_weights(self):
        all_weights=dict()
        all_weights['w1']=tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1']=tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        #输出层recontruction没有使用激活函数，将w2和b2都初始化为0
        all_weights['w2']=tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2']=tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights
    
    #计算损失函数及执行一部训练
    def partial_fit(self, X):
        #让session执行两个计算图
        cost, opt=self.sess.run((self.cost, self.optimizer), feed_dict={self.x:X, self.scale:self.training_scale})
        return cost

    #自编码器训练完毕后，在测试集上对模型性能进行评测
    def calc_total_cost(self,X):
        return self.sess.run(self.cost, feed_dict={self.x:X, self.scale:self.training_scale})

    #返回自编码器隐含层的输出结果
    #获取抽象后的特征
    #自编码器隐含层的主要功能就是学习出数据中的高阶特征
    def transform(self,X):
        return self.sess.run(self.hidden, feed_dict={self.x:X, self.scale:self.training_scale})

    #将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据
    def generate(self, hidden=None):
        if hidden is None:
            hidden=np.random_normal(size=self.weights['b1'])
        return self.sess.run(slef.reconstruction, feed_dict={self.hidden:hidden})

    #输入数据是原始数据，输出数据是复原后的数据
    #效果等同于上面的transform和generate
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction, feed_dict={self.x:X, self.scale:self.training_scale})

    #返回隐含层的权重
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    #返回隐含层的偏置
    def getBiases(self):
        return self.sess.run(self.weights['b1'])




mnist=input_data.read_data_sets(data_path, one_hot=True)

#对训练数据和测试数据进行标准化处理(深度学习也需要数据标准化处理？)
def standard_scale(X_train, X_test):
    preprocessor=prep.StandardScaler().fit(X_train)
    X_train=preprocessor.transform(X_train)
    X_test=preprocessor.transform(X_test)
    return X_train,X_test
#获取随机数据block
def get_random_block_from_data(data, batch_size):
    start_index=np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index +  batch_size)]

X_train,X_test=standard_scale(mnist.train.images, mnist.test.images)

n_samples=int(mnist.train.num_examples)
training_epochs=20
batch_size=128
display_step=1


autoencoder=AdditiveGaussianNoiseAutoencoder(n_input=784, n_hidden=200, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(learning_rate=0.001), scale=0.01)

for epoch in range(training_epochs):
    avg_cost=0.0
    total_batch=int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs=get_random_block_from_data(X_train, batch_size)
        cost=autoencoder.partial_fit(batch_xs)
        avg_cost+=cost/n_samples*batch_size
    if epoch % display_step==0:
        print("Epoch:", "%04d" % (epoch+1), " cost=", "{:.9f}".format(avg_cost) )

print("Total cost:" + str(autoencoder.calc_total_cost(X_test)) )


#自编码器作为一种无监督学习方法，它与其他无监督学习的主要区别在于，它不是对数据进行聚类，而是提取其中最常用、最频繁出现的高阶特征，根据这些高阶特征重构数据
