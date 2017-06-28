# -*- coding:utf8 -*-

import tensorflow as tf
import numpy as np

X=[1,2]

state=[0.0, 0.0]

w_cell_state=np.asarray([[0.1, 0.2],[0.3, 0.4]])
w_cell_input=np.asarray([0.5, 0.6])
b_cell=np.asarray([0.1, -0.1])

"""
用于输出的全连接层参数
"""
w_output=np.asarray([[1.0],[2.0]])
b_output=0.1


for i in range(len(X)):
    before_activation=np.dot(state, w_cell_state) + X[i]*w_cell_input + b_cell
    state=np.tanh(before_activation)

    final_output=np.dot(state, w_output) + b_output

    print("before activation:", before_activation)
    print("state:", state)
    print("output:", final_output)
