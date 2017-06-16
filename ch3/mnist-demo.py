#-*-coding:utf8-*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

data_path="/usr/bigdata/data/mnist_data/"

mnist=input_data.read_data_sets(data_path, one_hot=True)

print("shape of train data:")
print(mnist.train.images.shape, mnist.train.labels.shape)
print("shape of test data:")
print(mnist.test.images.shape, mnist.test.labels.shape)
print("shape of validation data:")
print(mnist.validation.images.shape, mnist.validation.labels.shape)

#将这个session注册为默认的session
#之后的运算也默认跑在这个session中
#不同的session之间的数据和运算都是相互独立的
sess=tf.InteractiveSession()

#数据输入的地方
#第一个参数是数据类型
#第二个参数代表tensor的shape，也就是数据的尺寸，None代表不限制数据的条数，784表示每条数据是一个784维的向量
x=tf.placeholder(tf.float32, [None, 784])

#Variable对象是用来存储模型参数的
#不同于存储数据的tensor，它一旦使用掉就会消失
#而Variable在模型训练迭代中式持久化的，它可以长期存在，并且在每轮迭代中被更新
#784表示特征的维数，10表示有10类（数字0-9，所以是10类）
W=tf.Variable(tf.zeros([784, 10]))
b=tf.Variable(tf.zeros([10]))


y=tf.nn.softmax(tf.matmul(x, W)+b)

#真实的label
y_=tf.placeholder(tf.float32, [None, 10])

#为了训练模型，需要定义一个损失函数loss function（cost function），来描述模型对问题的分类精度
#训练的目的是不断将这个loss减少，直到到达一个全局最优或者局部最优
#对于多分类问题，常使用交叉熵cross entropy作为损失函数loss function
#tf.reduce_sum求和
#tf.reduce_mean对每个batch数据结果求平均值
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

#使用随机梯度下降SGD
#学习率设置为0.5
#优化目标设定为上面定义的cross_entropy
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#全局参数初始化器
tf.global_variables_initializer().run()

#迭代地执行训练操作train_step
#每次都随机地从训练集中抽取100条样本构成一个mini-batch，并feed给placeholder，然后调用train_step对这些样本进行训练
for i in range(1000):
    batch_xs, batch_ys=mnist.train.next_batch(100)
    train_step.run({x:batch_xs, y_:batch_ys})
##############################################################模型训练完成

##############################################################对模型的准确率进行验证

#tf.argmax从一个tensor中寻找最大值的序号
#tf.equal返回预测类别和真实类别相同的，返回bool值
#tf.argmax(y,1) 每个样本预测返回10维的向量，找出其中最大值，也就是概率最大的，作为该样本的类别
correct_prediction=tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#tf.cast将bool值转换为float32
#再求平均值
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#这个准确率验证过程比较奇怪
#通常我是这么认为的：模型训练完成，输出模型的权重w和偏置bias b
#然后将测试数据代入到y=wx+b中，算出预测结果
#再把测试数据的真实结果与预测结果进行比较
#从而得出准确率
#而这个测试验证过程似乎跟想象的不一样
print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))



########################################################################总结
#整个流程：
#1.定义算法公式，也就是神经网络forward前馈时的计算
#2.定义损失函数loss function，选定优化器，并指定优化器优化loss
#3.迭代地对数据进行训练
#4.在测试集或者验证集上对准确率进行评测

#注意：
#tensorflow和spark类似，定义的各个公式其实只是计算图computation graph
#在执行代码时候，计算还没有实际发生，只有等调用run方法，并且feed数据时候，计算才真正执行
