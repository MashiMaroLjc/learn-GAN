import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.examples.tutorials.mnist import input_data

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
Mnist = input_data.read_data_sets('MNIST_data/', one_hot=False)

image_size = 28 * 28
z_size = 100
batch_size = 64
max_iter = 10000
log_i = 10
save_i = 1000
verbose = True
resotre = True
model_path = "my_net/GAN_net.ckpt"


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def combine(image):
    if len(image) > 9:
        image = image[:9]
    rows = []
    for i in range(3):
        cols = []
        for j in range(3):
            index = i * 3 + j
            img = image[index].reshape(28, 28)
            cols.append(img)
        row = np.concatenate(tuple(cols), axis=0)
        rows.append(row)
    new_image = np.concatenate(tuple(rows), axis=1)
    return new_image


def G(z, reuse=False):
    with tf.variable_scope("G", reuse=reuse):
        layer1 = tf.layers.dense(z, 512, activation=tf.nn.relu)
        layer2 = tf.layers.dense(layer1, 512, activation=tf.nn.relu)
        layer3 = tf.layers.dense(layer2, image_size, activation=tf.nn.tanh)
    return layer3


def D(x, reuse=False):
    with tf.variable_scope("D", reuse=reuse):
        layer1 = tf.layers.dense(x, 512, activation=tf.nn.relu)
        layer2 = tf.layers.dense(layer1, 512, activation=tf.nn.relu)
        layer3 = tf.layers.dense(layer2, 1)
    return layer3


x = tf.placeholder(tf.float32, shape=[None, image_size], name="image")
z = tf.placeholder(tf.float32, shape=[None, z_size], name="z")
g_net = G(z)
d_net_real = D(x)
d_net_fake = D(g_net, reuse=True)

vars = tf.trainable_variables()

D_PARAMS = [var for var in vars if var.name.startswith("D")]
G_PARAMS = [var for var in vars if var.name.startswith("G")]
# d_clip = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in D_PARAMS]
# d_clip = tf.group(*d_clip)  # 限制参数
d_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_net_fake, labels=tf.zeros_like(d_net_fake))) + tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_net_real, labels=tf.ones_like(d_net_real)))
# d_loss = tf.reduce_mean(d_net_fake) - tf.reduce_mean(d_net_real)

d_opt = tf.train.RMSPropOptimizer(1e-4).minimize(
    d_loss,
    global_step=tf.Variable(0),
    var_list=D_PARAMS
)

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_net_fake, labels=tf.ones_like(d_net_fake)))
# g_loss = tf.reduce_mean(-d_net_fake)
g_opt = tf.train.RMSPropOptimizer(1e-4).minimize(
    g_loss,
    global_step=tf.Variable(0),
    var_list=G_PARAMS
)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
if resotre:
    print("Load Model ...")
    saver.restore(sess,model_path)
for it in range(max_iter):
    X = Mnist.train.next_batch(batch_size)[0]
    image = (X - 0.5) / 0.5
    noise = np.random.normal(size=[batch_size, z_size])
    for _ in range(1):
        _,  d_loss_value = sess.run([d_opt,d_loss], feed_dict={
            x: image,
            z: noise
        })
    for _ in range(2):
        _, g_loss_value = sess.run([g_opt, g_loss], feed_dict={
            z: noise
        })
    if it == 0 or (it + 1) % log_i == 0:
        print("[{}/{}] d_loss:{:.5f}  g_loss:{:.5f}".format(it + 1, max_iter, d_loss_value, g_loss_value))
    if verbose:
        noise = np.random.normal(size=[batch_size, z_size])
        fake_image = sess.run([g_net], feed_dict={
            z: noise
        })[0]
        if len(fake_image) >= 9:
            new_image = combine(fake_image)
            # print("new image max:{}   min:{}".format(new_image.max(),new_image.min()))
            new_image = np.clip(new_image * 0.5 + 0.5, 0, 1)  # cv2.imshow 需要0-1浮点，或0~255int
            w, h = new_image.shape
            new_image = cv2.resize(new_image, (int(h * 3), int(w * 3)))
            cv2.imshow("mnist example", new_image)
            cv2.waitKey(1)
            if (it + 1) % save_i == 0:
                cv2.imwrite("mnist_example.jpg", new_image)

                save_path = saver.save(sess,model_path )
                print("Model save in %s" % save_path)
        else:
            print("image too less")
