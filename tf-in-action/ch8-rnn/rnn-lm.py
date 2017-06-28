# -*- coding:utf8 -*-

import tensorflow as tf
from tensorflow.models.rnn.ptb import reader

DATA_PATH="/usr/bigdata/data/PTB/simple-examples/data/"

train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

print(len(train_data))
print(train_data[:100])
