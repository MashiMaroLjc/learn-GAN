# coding:utf-8

import numpy as np
import tensorflow as tf
import time
from sklearn.preprocessing import PolynomialFeatures
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

W = 28
H = 28  
chanel = 1

BATCH_SIZE = 16
K = 1.0
LAMBDA = 10  # 梯度norm在目标函数中的权重,大概训练过程是先将D满足梯度norm的目标，这个参数会影响达到这个目标的时间
Mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

is_restore = True


def preprocess(x):
    return x

def deprocess(x):
    image = x* 255
    return image.reshape([9,W,H,chanel]).astype("uint8")

def combine(image):
    assert len(image) >= 9
    if len(image) > 9:
        image = image[:9]
    rows = []
    for i in range(3):
        cols = []
        for j in range(3):
            index = i * 3 + j
            img = image[index].reshape(H, W)
            cols.append(img)
        row = np.concatenate(tuple(cols), axis=0)
        rows.append(row)
    new_image = np.concatenate(tuple(rows), axis=1)
    return new_image.astype("uint8")


def leakyRelu(x, alpha=0.2):
    output = tf.maximum(alpha * x, x)
    return output


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def flatten(inputs):
    dim = tf.reduce_prod(tf.shape(inputs)[1:])
    return tf.reshape(inputs, [-1, dim])


def conv2d(inputs, filter_shape, name, stride=(1, 1), padding="SAME", baies=True, act_fun=None):
    filter_w = tf.get_variable(name + ".w", shape=filter_shape,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
    if baies:
        filter_b = tf.get_variable(name + ".b", initializer=tf.zeros((1, filter_shape[-1])) + 0.1)
        feature_map = tf.nn.conv2d(inputs, filter_w, strides=[1, stride[0], stride[1], 1], padding=padding) + filter_b
    else:
        feature_map = tf.nn.conv2d(inputs, filter_w, strides=[1, stride[0], stride[1], 1], padding=padding)
    if act_fun:
        feature_map = act_fun(feature_map)
    return feature_map


def decov2d(inputs, filter_shape, output_shape, name, stride=(1, 1),
            padding="SAME", bn=False, baies=True, act_fun=None):
    def batch_normalization(inputs, out_size, name, axes=0):
        mean, var = tf.nn.moments(inputs, axes=[axes])
        scale = tf.get_variable(name=name + ".scale", initializer=tf.ones([out_size]))
        offset = tf.get_variable(name=name + ".shift", initializer=tf.zeros([out_size]))
        epsilon = 0.001
        return tf.nn.batch_normalization(inputs, mean, var, offset, scale, epsilon, name=name + ".bn")

    w = tf.get_variable(name + ".w", shape=[filter_shape[0], filter_shape[1], output_shape[-1], inputs.get_shape()[-1]],
                        initializer=tf.truncated_normal_initializer(stddev=0.02))
    if baies:
        b = tf.get_variable(name + ".b", [output_shape[-1]], initializer=tf.constant_initializer(0.01))
        convt = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape,
                                       strides=[1, stride[0], stride[1], 1], padding=padding) + b
    else:
        convt = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape,
                                       strides=[1, stride[0], stride[1], 1], padding=padding)
    if bn:
        convt = batch_normalization(flatten(convt), output_shape[1] * output_shape[2] * output_shape[3],
                                    name=name + ".bn")
        convt = tf.reshape(convt, [-1] + output_shape[1:])
    if act_fun:
        convt = act_fun(convt)
    return convt


def dense(inputs, shape, name, bn=False, act_fun=None):
    W = tf.get_variable(name + ".w", initializer=tf.random_normal(shape=shape) / np.sqrt(shape[0] / 2))
    b = tf.get_variable(name + ".b", initializer=(tf.zeros((1, shape[-1])) + 0.1))
    y = tf.add(tf.matmul(inputs, W), b)

    def batch_normalization(inputs, out_size, name):
        mean, var = tf.nn.moments(inputs, axes=[0])
        scale = tf.get_variable(name=name + ".scale", initializer=tf.ones([out_size]))
        offset = tf.get_variable(name=name + ".shift", initializer=tf.zeros([out_size]))
        epsilon = 0.001
        return tf.nn.batch_normalization(inputs, mean, var, offset, scale, epsilon, name=name + ".bn")

    if bn:
        y = batch_normalization(y, shape[1], name=name + ".bn")
    if act_fun:
        y = act_fun(y)
    return y


def max_pooling(inputs, ksize=(2, 2), stride=(2, 2), padding="SAME"):
    return tf.nn.max_pool(inputs,
                          ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding=padding)

hidden = 16

def D(inputs, condition,name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        # l1 = conv2d(tf.reshape(inputs, [-1, W, H, chanel]), [5, 5, chanel, hidden], name="cov1",
        #             act_fun=leakyRelu) 
        # l2 = conv2d(l1, [5, 5, hidden,int(hidden/2)], name="cov2", 
        #             act_fun=leakyRelu) 
        # # l3 = conv2d(l2, [5, 5, int(hidden/2), int(hidden/4)], name="cov3", 
        # #             act_fun=leakyRelu) # W/8 H/8 HIDDEN*4
        # l3 = flatten(l2)
        # l4 = dense(l3, [int(W)*int(H)*int(hidden/2) ,128], name="feature")
        # l5 = tf.concat([l4,condition], axis=1)
        # l6 = dense(l5, [128+10 ,64], name="feature2")
        # y = dense(l6, [64, 1], name="output")
        inputs = tf.concat([inputs,condition], axis=1)
        l1 = dense(inputs, [W*H*chanel+10, 256], name="ly1",act_fun=leakyRelu)
        l1 = dense(l1, [256,128], name="ly2",act_fun=leakyRelu)
        l1 = dense(l1, [128,64], name="ly3",act_fun=leakyRelu)
        y = dense(l1, [64,1], name="output")
        return y


def G(inputs,condition,name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        inputs = tf.concat([inputs,condition], axis=1)
        l1 = dense(inputs, [100+10, 128], name="ly1",act_fun=tf.nn.relu)
        l1 = dense(l1, [128,512], name="ly2",act_fun=tf.nn.relu)
        l1 = dense(l1, [512,1024], name="ly3",act_fun=tf.nn.relu)
        y = dense(l1, [1024,W*H*chanel], name="output",act_fun=tf.nn.sigmoid)
        # inputs = tf.concat([inputs,condition], axis=1)
        # l1 = dense(inputs, [100+9, W* H * hidden], name="linear", bn=True,act_fun=tf.nn.relu)
        # l2 = decov2d(tf.reshape(l1, [-1, W, H, hidden]), filter_shape=[5, 5], output_shape=[BATCH_SIZE, W, H, hidden],
        #         name="decov", bn=True,act_fun=tf.nn.relu)
        # l3 = decov2d(tf.reshape(l2, [-1, W, H, hidden]), filter_shape=[5, 5], output_shape=[BATCH_SIZE, W, H, hidden],
        #          name="decov2",bn=True,act_fun=tf.nn.relu)
        # y = decov2d(tf.reshape(l3, [-1, W, H, hidden]), filter_shape=[5, 5], output_shape=[BATCH_SIZE, W, H, chanel],
        #         name="decov3",act_fun=tf.nn.sigmoid)
        # y = flatten(y)
        return y


z = tf.placeholder(tf.float32, [None, 100], name="noise")  # 100
x = tf.placeholder(tf.float32, [None, W * H * chanel], name="image")  # 36*36*3
c = tf.placeholder(tf.float32, [None, 10], name="condition")  # 36*36*3

real_out = D(x, c,"D")
gen = G(z,c,"G")
fake_out = D(gen,c,"D", reuse=True)

vars = tf.trainable_variables()

D_PARAMS = [var for var in vars if var.name.startswith("D")]
G_PARAMS = [var for var in vars if var.name.startswith("G")]

wd = tf.reduce_mean(real_out) - tf.reduce_mean(fake_out)
alpha = tf.random_uniform(
    shape=[BATCH_SIZE, 1],
    minval=0.,
    maxval=1.
)  # 采样
d_loss = tf.reduce_mean(fake_out) - tf.reduce_mean(real_out)

# 插值
insert_value = gen - alpha * (x - gen)

gradients = tf.gradients(D(insert_value,c,name="D", reuse=True), [insert_value])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))  # 求范数
gradient_penalty = tf.reduce_mean((slopes - K) ** 2)  # 最少化这个会使梯度集中在K值附近


l1_loss = tf.reduce_mean(tf.abs(gen-x))


d_loss += LAMBDA * gradient_penalty
g_loss = tf.reduce_mean(-fake_out)

d_opt = tf.train.AdamOptimizer(1e-4, beta1=0.4, beta2=0.9).minimize(
    d_loss,
    global_step=tf.Variable(0),
    var_list=D_PARAMS
)

g_opt = tf.train.AdamOptimizer(1e-4, beta1=0.4, beta2=0.9).minimize(
    g_loss,
    global_step=tf.Variable(0),
    var_list=G_PARAMS
)



model_path = "my_net2/GAN_net.ckpt"

b = 0
e = 100 * 1000
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if is_restore:
    saver = tf.train.Saver()
    # 提取变量
    saver.restore(sess, model_path)
    print("Model restore...")


print("Training……")
CRITICAL_NUM = 5
epoch = 100
length = Mnist.train.num_examples
for ep in range(epoch):
    bt = time.time()
    step = 0
    while step < length:
        critical_num = CRITICAL_NUM
        for _ in range(critical_num):
            noise = np.random.normal(size=(BATCH_SIZE, 100))
            batch_xs,batch_ys= Mnist.train.next_batch(BATCH_SIZE)
            batch_xs = preprocess(batch_xs)
            _, d_loss_v,wd_v = sess.run([d_opt, d_loss,wd], feed_dict={
                x: batch_xs,
                c: batch_ys,
                z: noise,
            })
            step += BATCH_SIZE
            print("\rep: %d/%d   batch: %d/%d  loss:%.7f  "%(ep+1,epoch,step,length,wd_v),end="")
                
        for _ in range(1):
            noise = np.random.normal(size=(BATCH_SIZE, 100))
            batch_xs,batch_ys = Mnist.train.next_batch(BATCH_SIZE)
            batch_xs = preprocess(batch_xs)
            _, g_loss_v = sess.run([g_opt, g_loss], feed_dict={
                x: batch_xs,
                c:batch_ys,
                z: noise,
            })
            step += BATCH_SIZE
            print("\rep: %d/%d   batch: %d/%d  loss:%.7f  "%(ep+1,epoch,step,length,wd_v),end="")


    print()
    et = time.time()
    noise = np.random.normal(size=(9, 100))
    noise_y = np.array([
        [0,0,0,0,0,0,0,0,0,1], 
        [0,0,0,0,0,0,0,0,1,0], 
        [0,0,0,0,0,0,0,1,0,0], 
        [0,0,0,0,0,0,1,0,0,0], 
        [0,0,0,0,0,1,0,0,0,0], 
        [0,0,0,0,1,0,0,0,0,0], 
        [0,0,0,1,0,0,0,0,0,0], 
        [0,0,1,0,0,0,0,0,0,0], 
        [0,1,0,0,0,0,0,0,0,0], 
        ])
    generate = sess.run(gen, feed_dict={
        z: noise,
        c:noise_y
    })
    generate = deprocess(generate)
    image = combine(generate)
    Image.fromarray(image).save("image2/ep_%d.jpg" % (ep + 1))
    saver = tf.train.Saver()
    save_path = saver.save(sess, model_path)
    print("Model save in %s" % save_path)
    print("Cost time: %d s"%(et-bt))
    print("-----------------------------")
