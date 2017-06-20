#-*-coding:utf8-*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data_path="/usr/bigdata/data/mnist_data/"

mnist=input_data.read_data_sets(data_path, one_hot=True)

#创建一个tf默认的session，这样，后面执行各种操作就无须指定session
sess=tf.InteractiveSession()

#输入层节点数
in_units=784
#隐含层节点数
h1_units=300
#隐含层权重
#初始化为截断的正态分布
#因为模型使用的激活函数是ReLU，所以需要使用正态分布给参数加一点噪声，来打破完全对称并且避免0梯度。
W1=tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
#在一些模型中，有时还需要给偏置赋上一些小的非零值来避免死亡神经元dead neuron
b1=tf.Variable(tf.zeros([h1_units]))

#输出层权重
W2=tf.Variable(tf.zeros([h1_units, 10]))
b2=tf.Variable(tf.zeros([10]))

#输入
x=tf.placeholder(tf.float32, [None, in_units])
#Dropout的比率
#通常，训练时小于1，预测时等于1
keep_prob=tf.placeholder(tf.float32)


#第一步：定义模型网络结构
#定义隐含层，使用ReLU激活函数
hidden1=tf.nn.relu(tf.matmul(x, W1)+b1)
#使用Dropout
#随机将一部分节点置为0
hidden1_drop=tf.nn.dropout(hidden1, keep_prob)

#输出层
y=tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

#实际输出
y_=tf.placeholder(tf.float32, [None, 10])

#第二步：定义损失函数和优化器
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
#使用自适应优化器Adagrad
train_step=tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

#第三步：训练
tf.global_variables_initializer().run()
#因为加入了隐含层，需要更多的训练迭代来优化模型参数，以达到一个比较好的效果。
for i in range(3000):
    batch_xs, batch_ys=mnist.train.next_batch(100)
    #一般来说，对于越复杂，规模越大的神经网络，dropout的效果越显著
    #这里，保留75%的节点，其余的25%都置为0
    train_step.run({x: batch_xs, y_:batch_ys, keep_prob:0.75})

#第四步：对模型进行准确率评测
correct_prediction=tf.equal(tf.argmax(y,1) , tf.argmax(y_, 1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))


#准确率从92%提升到98%
#仅仅增加了一个隐含层
#使用了trick：dropout、adagrad、relu
#多层神经网络依靠隐含层，可以组合出高阶特征，比如横线、竖线、圆圈等，之后将这些高阶特征或者说组件组合成数字，就能实现精准匹配和分类
