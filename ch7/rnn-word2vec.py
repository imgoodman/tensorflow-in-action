# -*- coding:utf8 -*-
import tensorflow as tf
import zipfile
import os
import collections
import numpy as np
import random
import math

url="http://mattmahoney.net/dc/"
file_name=url+"text8.zip"
data_path="/usr/bigdata/data/word2vec-data/text8.zip"

def maybe_download(filename, expected_bytes):
    if not os.path:
        pass

def read_data(filepath):
    f=zipfile.ZipFile(filepath)
    #print(f.namelist())
    #f.read(f.namelist()[0])
    data=tf.compat.as_str(f.read(f.namelist()[0]))
    #print(data[:17005207].split())
    #data=data[:17005207]
    data=data[:17005207*4].split()
    return data

words=read_data(data_path)
print("data size:", len(words))


vocabulary_size=50000

def build_dataset(words):
    count=[['UNK',-1]]
    count.extend( collections.Counter(words).most_common(vocabulary_size-1)  )
    #print(count[:20])
    dictionary=dict()
    for word,_ in count:
        dictionary[word]=len(dictionary)

    data=list()
    unk_count=0
    for word in words:
        if word in dictionary:
            index=dictionary[word]
        else:
            index=0
            unk_count += 1
        data.append(index)
    count[0][1]=unk_count
    reverse_dictionary=dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary=build_dataset(words)

del words
#print("most top 5 common words:", count[:5])
#print("sample data",data[:5])
#for i in data[:10]:
#    print(reverse_dictionary[i])
#build_dataset(words)


data_index=0
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips ==0
    assert num_skips <= 2*skip_window
    batch=np.ndarray(shape=(batch_size), dtype=np.int32)
    labels=np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span=2*skip_window+1
    buffer=collections.deque(maxlen=span)


    for _ in range(span):
        buffer.append(data[data_index])
        data_indx=(data_index+1)%len(data)
    for i in range(batch_size//num_skips):
        target=skip_window
        targets_to_avoid=[skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target=random.randint(0, span-1)
            targets_to_avoid.append(target)
            batch[i*num_skips+j]=buffer[skip_window]
            labels[i*num_skips+j, 0]=buffer[target]
        buffer.append(data[data_index])
        data_index=(data_index+1) % len(data)
    return batch,labels

#batch,labels=generate_batch(batch_size=8, num_skips=2, skip_window=1)
#for i in range(8):
#    print(batch[i],labels[i])
#    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i,0], reverse_dictionary[labels[i,0]])



batch_size=128
embedding_size=128
skip_window=1
num_skips=2

valid_size=16
valid_window=100
valid_examples=np.random.choice(valid_window, valid_size, replace=False)
num_sampled=64


graph=tf.Graph()
with graph.as_default():
    train_inputs=tf.placeholder(tf.int32, shape=[batch_size])
    train_labels=tf.placeholder(tf.int32, shape=[batch_size,1])
    valid_dataset=tf.constant(valid_examples, dtype=tf.int32)
    with tf.device("/cpu:0"):
        embeddings=tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], -1.0, 1.0))
        embed=tf.nn.embedding_lookup(embeddings, train_inputs)
        nce_weights=tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
        nce_biases=tf.Variable(tf.zeros([vocabulary_size]))
    loss=tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size))

    optimizer=tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm=tf.sqrt(tf.reduce_mean(tf.square(embeddings),1,keep_dims=True))
    normalized_embeddings=embeddings/norm
    valid_embeddings=tf.nn.embedding_lookup(normalize_embeddings, valid_dataset)
    similarity=tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    init=tf.global_variables_initializer()

num_step=100001

with tf.Session(graph=graph) as session:
    init.run()
    print("initialized")
    average_loss=0
    for step in range(num_steps):
        batch_inputs, batch_labels=generate_batch(batch_size, num_skips, skip_window)
        _,loss_val=sess.run([optimizer, loss], feed_dict={train_inputs: batch_inputs, train_labels:batch_labels})
        average_loss+=loss_val
        if step%2000==0:
            if step>0:
                average_loss/=2000
            print("average loss at step: %d is %g" % (step, average_loss))
            average_loss=0
        if step%10000==0:
            sim=similarity.eval()
            for i in range(valid_size):
                valid_word=reverse_dictionary[valid_exmaples[i]]
                top_k=8
                nearest=(-sim[i,:]).argsort()[1:top_k+1]
                log_str="nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word=reverse_dictionary[nearest[k]]
                    log_str="%s %s" % (log_str, close_word)
                print(log_str)
    final_embeddings=normalized_embeedings.eval()
