import tensorflow as tf
import numpy as np
from gan.network import Network
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
import cv2

z_dim = 64

class DCGAN(Network):
    def __init__(self, X_size, Y_size , batch_size = 100, trainable=True, name=None):
        self.name = name
        self.inputs = []
        self.dataX = tf.placeholder(tf.float32, shape = [None] + X_size)
        self.dataZ = tf.placeholder(tf.float32, shape = [None] + Y_size)
        self.layers = {'dataX':self.dataX, 'dataZ':self.dataZ}
        self.trainable = trainable
        self.batch_size = batch_size
        self.setup()
        self.saver = tf.train.Saver()

    def setup(self):
        #generator
        (self.feed('dataZ')
         .fc(7*7*128, name='gen_proj')
         .reshape([-1, 7, 7, 128], name='gen_reshape')
         .deconv(5, 5, 64, 2, 2, relu=False, name='gen_deconv1')
         .batch_norm(name='gen_deconv1_bn')
         .relu(name='gen_deconv1_relu')
         .deconv(5, 5, 1, 2, 2, relu=False, name='gen_deconv2')
         .tanh(name='gen_deconv2_tanh')
         .rename(name='gen_prob'))


        #discriminator
        (self.feed('dataX')
         .reshape([-1, 28, 28, 1], name='dis_reshape')
         .conv(5, 5, 64, 1, 1, relu=False, name='dis_conv1')
         .batch_norm(name='dis_conv1_bn')
         .lrelu(name='dis_conv1_lrelu')
         .conv(5, 5, 128, 2, 2, relu=False, name='dis_conv2')
         .batch_norm(name='dis_conv2_bn')
         .lrelu(name='dis_conv2_lrelu')
         .reshape([-1, 7 * 7 * 128], name='dis_reshape2')
         .fc(1, name='dis_prob', relu=False)
         .rename(name='real_dis'))

        (self.feed('gen_prob')
         .reshape([-1, 28, 28, 1], name='dis_reshape')
         .conv(5, 5, 64, 1, 1, relu=False, name='dis_conv1', reuse=True)
         .batch_norm(name='dis_conv1_bn', reuse=True)
         .lrelu(name='dis_conv1_lrelu')
         .conv(5, 5, 128, 2, 2, relu=False, name='dis_conv2', reuse=True)
         .batch_norm(name='dis_conv2_bn', reuse=True)
         .lrelu(name='dis_conv2_lrelu')
         .reshape([-1, 7 * 7 * 128], name='dis_reshape2')
         .fc(1, name='dis_prob', relu=False, reuse=True)
         .rename(name='gen_dis'))


from tensorflow.examples.tutorials.mnist import input_data
import os

mnist = input_data.read_data_sets("../dataset/MNIST/", one_hot=True)

learning_rate = 0.001
training_iters = 100000
batch_size = 16
display_step = 10
d_steps = 5


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples, step):
    for i, sample in enumerate(samples):
        cv2.imwrite('/home/tongth/git/tensorflow/test/' + str(step) + '_' + str(i) + '.png', sample.reshape(28,28) * 255)


def train(net, sess):
    sess.run(init)
    if os.path.exists("../learn_param/dcgan2.ckpt.meta"):
        net.load("../learn_param/dcgan2.ckpt", sess, True)

    step = 1
    while step < training_iters:
        D_loss_curr = None
        for _ in range(d_steps):
            X_mb, _ = mnist.train.next_batch(batch_size)
            z_mb = sample_z(batch_size, z_dim)

            _, D_loss_curr = sess.run(
                [net.opt_dis, net.cost_dis],
                feed_dict={net.dataX: X_mb, net.dataZ: z_mb}
            )

        X_mb, _ = mnist.train.next_batch(batch_size)

        _, G_loss_curr = sess.run(
            [net.opt_gen, net.cost_gen],
            feed_dict={net.dataX: X_mb, net.dataZ: sample_z(batch_size, z_dim)}
        )

        if step % 1000 == 0:
            print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'
                  .format(step, D_loss_curr, G_loss_curr))

            samples = sess.run(net.get_output('gen_prob'), feed_dict={net.dataZ: sample_z(16, z_dim)})
            plot(samples, step)

        step += 1

    save_path = net.saver.save(sess, "../learn_param/dcgan2.ckpt")
    print("Model saved infile : {}".format(save_path))

if __name__ == '__main__':
    net = DCGAN(X_size=[784], Y_size=[z_dim], batch_size=batch_size, name='dcgan')

    #gen_var = net.get_variables('gen_fc') + net.get_variables('gen_log_prob')
    gen_var = net.get_variables('gen_deconv1') + net.get_variables('gen_deconv2') + net.get_variables('gen_deconv1_bn') + net.get_variables('gen_proj')
    dis_var = net.get_variables('dis_prob') + net.get_variables('dis_conv1') + net.get_variables('dis_conv2') + net.get_variables('dis_conv2_bn') + net.get_variables('dis_conv1_bn')

    net.set_cost_gen(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net.get_output('gen_dis'), labels=tf.ones_like(net.get_output('gen_dis')))))
    net.set_optimizer_gen(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net.cost_gen, var_list=gen_var))

    dis_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net.get_output('real_dis'), labels=tf.ones_like(net.get_output('real_dis')))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net.get_output('gen_dis'), labels=tf.zeros_like(net.get_output('gen_dis'))))
    net.set_cost_dis(dis_cost)
    net.set_optimizer_dis(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net.cost_dis, var_list=dis_var))


    init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for i in range(10):
            train(net, sess)
            print("complete {} iteration".format(i+1))
