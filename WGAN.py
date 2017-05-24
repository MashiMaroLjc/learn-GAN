# coding:utf-8
"""
WGAN
- D的最后一层取消sigmoid
- 损失函数取消log
- D的w 取值限制在[-c,c]区间内
- 使用RMSProp或SGD并以较低的学习率进行优化
- WD 理论上可以衡量WGAN的训练效果 (实验情况多次打脸)
项目文件:
MINIST_data minst数据，第一次运行该代码会自动从网络上下载
image 用于保存训练过程中生成的图片，自行创建或改代码路径
my_net 用于保存训练后的模型，自行创建或改代码路径
"""

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def combine(image):
    assert len(image) == 64
    rows = []
    for i in range(8):
        cols = []
        for j in range(8):
            index = i * 8 + j
            img = image[index].reshape(28, 28)
            cols.append(img)
        row = np.concatenate(tuple(cols), axis=0)
        rows.append(row)
    new_image = np.concatenate(tuple(rows), axis=1)
    return new_image.astype("uint8")





def dense(inputs, shape, name, bn=False, act_fun=None):
    W = tf.get_variable(name + ".w", initializer=tf.random_normal(shape=shape))
    b = tf.get_variable(name + ".b", initializer=(tf.zeros((1, shape[-1])) + 0.1))
    y = tf.add(tf.matmul(inputs, W), b)

    def batch_normalization(inputs, out_size, name, axes=0):
        mean, var = tf.nn.moments(inputs, axes=[axes])
        scale = tf.get_variable(name=name + ".scale", initializer=tf.ones([out_size]))
        offset = tf.get_variable(name=name + ".shift", initializer=tf.zeros([out_size]))
        epsilon = 0.001
        return tf.nn.batch_normalization(inputs, mean, var, offset, scale, epsilon, name=name + ".bn")

    if bn:
        y = batch_normalization(y, shape[1], name=name + ".bn")
    if act_fun:
        y = act_fun(y)
    return y


def D(inputs, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        l1 = dense(inputs, [784, 512], name="relu1", act_fun=tf.nn.relu)
        l2 = dense(l1, [512, 512], name="relu2", act_fun=tf.nn.relu)
        l3 = dense(l2, [512, 512], name="relu3", act_fun=tf.nn.relu)
        y = dense(l3, [512, 1], name="output")
        return y


def G(inputs, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        l1 = dense(inputs, [100, 512], name="relu1", act_fun=tf.nn.relu)
        l2 = dense(l1, [512, 512], name="relu2", act_fun=tf.nn.relu)
        l3 = dense(l2, [512, 512], name="relu3", act_fun=tf.nn.relu)
        y = dense(l3, [512, 784], name="output", bn=True, act_fun=tf.nn.sigmoid)
        return y


z = tf.placeholder(tf.float32, [None, 100], name="noise")  # 100
x = tf.placeholder(tf.float32, [None, 784], name="image")  # 28*28

real_out = D(x, "D")
gen = G(z, "G")
fake_out = D(gen, "D", reuse=True)

vars = tf.trainable_variables()

D_PARAMS = [var for var in vars if var.name.startswith("D")]
G_PARAMS = [var for var in vars if var.name.startswith("G")]

d_clip = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in D_PARAMS]
d_clip = tf.group(*d_clip)  # 限制参数

wd = tf.reduce_mean(real_out) - tf.reduce_mean(fake_out)
d_loss = tf.reduce_mean(fake_out) - tf.reduce_mean(real_out)
g_loss = tf.reduce_mean(-fake_out)

d_opt = tf.train.RMSPropOptimizer(1e-3).minimize(
    d_loss,
    global_step=tf.Variable(0),
    var_list=D_PARAMS
)

g_opt = tf.train.RMSPropOptimizer(1e-3).minimize(
    g_loss,
    global_step=tf.Variable(0),
    var_list=G_PARAMS
)
is_restore = False
# is_restore = True  # 是否第一次训练(不需要载入模型)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if is_restore:
    saver = tf.train.Saver()
    # 提取变量
    saver.restore(sess, "my_net/GAN_net.ckpt")
    print("Model restore...")


CRITICAL_NUM = 5
for step in range(100 * 1000):
    if step < 25 or step % 500 == 0:
        critical_num = 100
    else:
        critical_num = CRITICAL_NUM
    for ep in range(critical_num):
        noise = np.random.normal(size=(64, 100))
        batch_xs = mnist.train.next_batch(64)[0]
        _, d_loss_v, _ = sess.run([d_opt, d_loss, d_clip], feed_dict={
            x: batch_xs,
            z: noise
        })


    for ep in range(1):
        noise = np.random.normal(size=(64, 100))
        _, g_loss_v = sess.run([g_opt, g_loss], feed_dict={
            z: noise
        })
    print("Step:%d   D-loss:%.4f  G-loss:%.4f" % (step + 1, d_loss_v, g_loss_v))
    if step % 1000 == 999:
        batch_xs = mnist.train.next_batch(64)[0]
        # batch_xs = pre(batch_xs)
        noise = np.random.normal(size=(64, 100))
        mpl_v = sess.run(wd, feed_dict={
            x: batch_xs,
            z: noise
        })
        print("##################    Step %d  WD:%.4f ###############" % (step + 1, mpl_v))
        generate = sess.run(gen, feed_dict={
            z: noise
        })

        generate *= 255
        generate = np.clip(generate, 0, 255)
        image = combine(generate)
        Image.fromarray(image).save("image/Step_%d.jpg" % (step + 1))
        saver = tf.train.Saver()
        save_path = saver.save(sess, "my_net/GAN_net.ckpt")
        print("Model save in %s" % save_path)
sess.close()
