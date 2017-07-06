# -*- coding:utf8 -*-
import tensorflow as tf
import os
import numpy as np
from scipy import ndimage
from six.moves import cPickle as pickle

train_folder="/usr/bigdata/data/notMNIST_data/notMNIST_large/"
test_folder="/usr/bigdata/data/notMNIST_data/notMNIST_small/"

labels=["A","B","C","D","E","F","G","H","I","J"]

train_folders=[train_folder+label for label in labels]
test_folders=[test_folder+label for label in labels]

image_size=28
pixel_depth=255.0

def load_letter(folder, min_num_images):
    """
    加载指定字母letter目录下的所有照片
    返回(照片数量，照片的宽度，照片的高度)
    """
    image_files=os.listdir(folder)#列出指定目录下所有文件名，注意，仅仅是文件名称，不是文件的完整路径
    dataset=np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)#定义图片数据，三维数组：图片数量，图片宽度，图片长度
    print(folder)
    num_images=0#记录实际符合尺寸的照片数量，因为，有些照片尺寸不符合要求
    for image in image_files:#遍历每张照片
        image_file=os.path.join(folder, image)#照片的完整路径
        try:
            """
            scipy.ndimage.imread(fname, flatten=False, mode=None)
            imread uses the Python Imaging Library (PIL) to read an image
            read an image from a file as an array
            fname: str or file object, the file name or file object to be read
            flatten: bool, if True, flattens the color layers into a single gray-scale layer
            mode: str, mode to convert image to e.g. 'RGB'
            returns ndarray, the array obtained by reading the image
            """
            image_data = (ndimage.imread(image_file).astype(float) - pixel_depth/2) /pixel_depth#输入图片数据的正则化，均值为0， 同方差
            if image_data.shape!=(image_size, image_size):
                raise Exception("Unexpected image shape: %s" % str(image_data.shape))
            dataset[num_images, :, :]=image_data#三维数组的赋值
            num_images = num_images +1
        except IOError as e:
            print("Could not read:", image_file, ":", e, " - it\'s ok, skipping.")
    dataset=dataset[0:num_images, :, :]#只返回符合尺寸条件的照片array
    if num_images < min_num_images:
        raise Exception("Many fewer images than expected: %d < %d" % (num_images, min_num_images))
    print("Full dataset tensor:", dataset.shape)#实际照片数据的维度
    print("Mean:", np.mean(dataset))#实际照片的均值
    print("Standard deviation:", np.std(dataset))#实际照片的标准差
    return dataset#返回指定字母的照片数据


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    """
    加载指定目录下，所有字母的图片数据，并保存为.pickle文件。后续用到图片数据，直接读取这些.pickle文件即可，不必再去解析原始的照片了。
    返回每个字母的.pickle文件路径组成的array
    """
    dataset_names=[]#每个字母的.pickle路径组成的array
    for folder in data_folders:#遍历所有的字母目录
        set_filename=folder+".pickle"#某字母的图片数据.pickle的完整路径
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:#如果某字母的.pickle完整路径已经存在，则跳过
            print("%s already exist -- Skipping pickling." % set_filename)
        else:
            print("Pickling %s." % set_filename)
            dataset=load_letter(folder, min_num_images_per_class)#加载指定字母的所有图片数据array
            try:
                with open(set_filename,"wb") as f:#打开.pickle文件，写入模式
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)#将指定字母的图片array数据写到对应的.pickle文件中
            except Exception as e:
                print("Unable to save data to ", set_filename, ":", e)
    return dataset_names#返回由每个字母对应的.pickle文件的完整路径array




def make_arrays(nb_rows, img_size):
    """
    生成指定行数、大小的array
    """
    if nb_rows:
        dataset=np.ndarray(shape=(nb_rows, img_size, img_size), dtype=np.float32)
        labels=np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels=None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    """
    根据字母对应的.pickle文件路径，生成对应的训练数据和验证数据
    返回验证数据，验证标签，测试数据，测试标签
    后续调用,train_size:200000, valid_size=10000
    """
    num_classes=len(pickle_files)#pickle文件数量，就是类别的数量，A-J，10个分类
    valid_dataset, valid_labels=make_arrays(valid_size, image_size)#生成验证数据集和对应的标签（10000， 28， 28）
    train_dataset, train_labels=make_arrays(train_size, image_size)#生成训练数据集和对应的标签（200000， 28， 28）

    vsize_per_class = valid_size // num_classes#验证集中，每个类型的图片数量（向下取整），1000
    tsize_per_class = train_size // num_classes#训练集中，每个类型的图片数量（向下取整），20000

    start_v, start_t= 0, 0#验证数据、训练数据的索引起始点
    end_v, end_t = vsize_per_class, tsize_per_class#雁阵数据、训练数据的终止点索引，分别是1000， 20000

    end_l = vsize_per_class + tsize_per_class#21000

    """
    采用了enumerate
    label指代的是每个.pickle文件的索引
    """
    for label, pickle_file in enumerate(pickle_files):#enumerate将可迭代/可遍历的对象，组成一个索引序列，利用它可以同时获得缩影和值。多用于在for循环中得到计数
        try:
            with open(pickle_file, "rb") as f:#以只读的模式打开指定的.pickle文件
                letter_set = pickle.load(f)#读取类型的.pickle文件中的图片数据，（图片数量，图片宽度，图片高度）
                """
                modify a sequence in-place by shuffling its contents
                this function only shuffles the array along the first axis of a multi-dimensional array. the order of sub-arrays is changed but their contents remains the same（意思是，在每一列中，进行随机排序，而不是整体的数组）
                """
                np.random.shuffle(letter_set)
                """
                过程如下：
                加载某字母的数据集（例如，5万张照片）
                选取指定数量的验证集（例如，1000张照片）
                将它加入到整体的验证集合中
                更改验证数据集的起止索引
                选取指定数量的训练集（例如，20000张照片）
                将它加入到整体的训练数据集合中
                更改训练数据集的起止索引
                """
                if valid_dataset is not None:#如果需要验证集
                    valid_letter=letter_set[:vsize_per_class, :, :]#按照验证数据集数量要求，从该类型照片中取相应数量，[0:1000, :, :]
                    valid_dataset[start_v:end_v, :, :]=valid_letter#[0:1000]
                    valid_labels[start_v:end_v]=label#[0:1000]
                    start_v += vsize_per_class#1000
                    end_v += vsize_per_class#2000
                train_letter=letter_set[vsize_per_class:end_l, :, :]#[1000:21000]
                train_dataset[start_t:end_t,:,:]=train_letter#[0:20000]
                train_labels[start_t:end_t]=label#[0:20000]
                start_t += tsize_per_class#20000
                end_t += tsize_per_class#40000

        except Exception as e:
            print("Unable to process data from ", pickle_file, ":", e)
            raise
    return valid_dataset, valid_labels, train_dataset, train_labels



train_size=100000
valid_size=5000
test_size=5000

train_datasets=maybe_pickle(train_folders, 45000)#得到所有字母对应的.pickle文件完整路径
test_datasets=maybe_pickle(test_folders, 1800)#得到所有测试字母对应的.pickle文件完整路径

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
_,_,test_dataset, test_labels=merge_datasets(test_datasets, test_size)

print("Training:", train_dataset.shape, train_labels.shape)
print("Validation:", valid_dataset.shape, valid_labels.shape)
print("Testing:", test_dataset.shape, test_labels.shape)


pickle_file="/usr/bigdata/data/notMNIST_data/notMNIST.pickle"
try:
    with open(pickle_file, "wb") as f:
        save={"train_dataset": train_dataset, "train_labels": train_labels, "valid_dataset": valid_dataset, "valid_labels": valid_labels, "test_dataset": test_dataset, "test_labels": test_labels}
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print("Unable to save data to", pickle_file,":",e)
    raise

statinfo=os.stat(pickle_file)
print("Compressed pickle ", statinfo.st_size)
