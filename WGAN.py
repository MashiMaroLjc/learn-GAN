# coding:utf-8
"""
WGAN
- D的最后一层取消sigmoid
- 损失函数取消log
- D的w 取值限制在[-c,c]区间内
- 使用RMSProp或SGD并以较低的学习率进行优化
- MPL 理论上可以衡量WGAN的训练效果 (实验情况多次打脸)

项目文件:
MINIST_data minst数据，第一次运行该代码会自动从网络上下载
image 用于保存训练过程中生成的图片，自行创建或改代码路径
my_net 用于保存训练后的模型，自行创建或改代码路径

"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def conv2d(inputs, filter_shape, name, stride=(1, 1), padding="SAME"):
    filter_w = tf.get_variable(name + ".w", initializer=tf.random_normal(filter_shape))
    filter_b = tf.get_variable(name + ".b", initializer=tf.zeros((1, filter_shape[-1])) + 0.1)
    feature_map = tf.nn.conv2d(inputs, filter_w, strides=[1, stride[0], stride[1], 1], padding=padding) + filter_b
    return feature_map


def flatten(inputs):
    dim = tf.reduce_prod(tf.shape(inputs)[1:])
    return tf.reshape(inputs, [-1, dim])


def linear(inputs, shape, name):
    W = tf.get_variable(name + ".w", initializer=tf.random_normal(shape=shape))
    b = tf.get_variable(name + ".b", initializer=(tf.zeros((1, shape[-1])) + 0.1))
    y = tf.add(tf.matmul(inputs, W), b)
    return y


def max_pooling(inputs, ksize=(2, 2), stride=(2, 2), padding="SAME"):
    return tf.nn.max_pool(inputs,
                          ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding=padding)


def batch_normalization(inputs, out_size, name, axes=0):
    mean, var = tf.nn.moments(inputs, axes=[axes])
    scale = tf.get_variable(name=name + ".scale", initializer=tf.ones([out_size]))
    offset = tf.get_variable(name=name + ".shift", initializer=tf.zeros([out_size]))
    epsilon = 0.001
    return tf.nn.batch_normalization(inputs, mean, var, offset, scale, epsilon, name=name + ".bn")


def D(inputs, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        map1 = tf.nn.relu(linear(inputs, shape=[784, 64], name="relu1"))
        map2 = tf.nn.relu(linear(map1, shape=[64, 128], name="relu2"))
        map3 = tf.nn.relu(linear(map2, shape=[128, 512], name="relu3"))
        map4 = linear(map3, shape=[512, 1], name="output")
        return map4


def G(inputs, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        map1 = tf.nn.relu(linear(inputs, shape=[100, 64], name="relu1"))
        map2 = tf.nn.relu(linear(map1, shape=[64, 128], name="relu2"))
        map3 = tf.nn.relu(linear(map2, shape=[128, 256], name="relu3"))
        map4 = linear(map3, shape=[256, 784], name="output")
        map5 = batch_normalization(map4, 784, name="bn")
        map6 = tf.nn.sigmoid(map5)
        return map6


z = tf.placeholder(tf.float32, [None, 100], name="noise")  # 100
x = tf.placeholder(tf.float32, [None, 784], name="image")  # 28*28

real_out = D(x, "D")
gen = G(z, "G")
fake_out = D(gen, "D", reuse=True)

vars = tf.trainable_variables()
D_PARAMS = [var for var in vars if var.name.startswith("D")]
G_PARAMS = [var for var in vars if var.name.startswith("G")]

d_clip = [tf.assign(var, tf.clip_by_value(var, -0.02, 0.02)) for var in D_PARAMS]
d_clip = tf.group(*d_clip)  # 限制参数

mpl = tf.reduce_mean(real_out) - tf.reduce_mean(fake_out)
d_loss = tf.reduce_mean(fake_out) - tf.reduce_mean(real_out)
g_loss = -tf.reduce_mean(fake_out)

d_opt = tf.train.RMSPropOptimizer(5e-5).minimize(
    d_loss,
    global_step=tf.Variable(0),
    var_list=D_PARAMS
)

g_opt = tf.train.RMSPropOptimizer(5e-5).minimize(
    g_loss,
    global_step=tf.Variable(0),
    var_list=G_PARAMS
)


# is_restore = False
is_restore = True  # 是否第一次训练(不需要载入模型)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if is_restore:
    saver = tf.train.Saver()
    # 提取变量
    saver.restore(sess, "my_net/GAN_net.ckpt")
    print("Model restore...")

for step in range(30000):
    for ep in range(5):
        batch_xs, batch_ys = mnist.train.next_batch(64)
        noise = np.random.normal(size=(64, 100))
        _, d_loss_v, _ = sess.run([d_opt, d_loss, d_clip], feed_dict={
            x: batch_xs,
            z: noise
        })
        print("Step:%d  epoch:%d  Train D loss:%.4f " % (step + 1, ep + 1, d_loss_v))

    for ep in range(1):
        noise = np.random.normal(size=(64, 100))
        _, g_loss_v = sess.run([g_opt, g_loss], feed_dict={
            z: noise
        })
        # if ep % 10 == 9:
        print("Step:%d  epoch:%d  Train G loss:%.4f " % (step + 1, ep + 1, g_loss_v))
    if step % 500 == 499:
        batch_xs, batch_ys = mnist.train.next_batch(1)
        noise = np.random.normal(size=(1, 100))
        mpl_v = sess.run(mpl, feed_dict={
            x: batch_xs,
            z: noise
        })
        generate = sess.run(gen, feed_dict={
            z: noise
        })
        image = generate.reshape([28, 28])
        plt.imshow(image, cmap='gray')
        plt.savefig("image/Step_%d.jpg" % (step + 1))
        print("##################    Step %d  MPL:%.4f ###############" % (step + 1, mpl_v))
        saver = tf.train.Saver()
        save_path = saver.save(sess, "my_net/GAN_net.ckpt")
        print("Model save in %s" % save_path)
sess.close()
