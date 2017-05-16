import tensorflow as tf
import numpy as np
from gan.network import Network
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

z_dim = 100

class DCGAN(Network):
    def __init__(self, X_size, Y_size , trainable=True, name=None):
        self.name = name
        self.inputs = []
        self.dataX = tf.placeholder(tf.float32, shape = [None] + X_size)
        self.dataZ = tf.placeholder(tf.float32, shape = [None] + Y_size)
        self.layers = {'dataX':self.dataX, 'dataZ':self.dataZ}
        self.trainable = trainable
        self.setup()
        self.saver = tf.train.Saver()

    def setup(self):
        #generator
        (self.feed('dataZ')
         .reshape([-1,1,1,100], name='gen_reshape')
         .deconv(4,4, 1024,1,1, name='gen_deconv1')
         .deconv(5,5, 512, 2,2, relu= False, name='gen_deconv2')
         .bn(512, name='gen_deconv2_bn')
         .relu(name='gen_deconv2_relu')
         .deconv(5, 5, 256, 2, 2, relu= False, name='gen_deconv3')
         .bn(256, name='gen_deconv3_bn')
         .relu(name='gen_deconv3_relu')
         .deconv(5, 5, 128, 2, 2, relu=False, name='gen_deconv4')
         .bn(128, name='gen_deconv4_bn')
         .relu(name='gen_deconv4_relu')
         .deconv(5, 5, 3, 2, 2, relu=False, name='gen_deconv5')
         .tanh(name='gen_deconv3_relu')
         .rename(name='gen_prob'))


        #discriminator
        (self.feed('dataX')
         .conv(5,5,128,2,2, relu= False,name='dis_conv1')
         .lrelu(name='dis_conv1_lrelu')
         .conv(5,5,256,2,2, relu= False, name='dis_conv2')
         .bn(256, name='dis_conv2_bn')
         .lrelu(name='dis_conv2_lrelu')
         .conv(5, 5, 512, 2, 2, relu= False, name='dis_conv3')
         .bn(512, name='dis_conv3_bn')
         .lrelu(name='dis_conv3_lrelu')
         .conv(5, 5, 1024, 2, 2, relu= False, name='dis_conv4')
         .bn(1024, name='dis_conv4_bn')
         .lrelu(name='dis_conv4_lrelu')
         .conv(4,4, 1, 4, 4, relu= False, name='dis_prob')
         .sigmoid(name='real_dis'))

        (self.feed('gen_prob')
         .conv(5, 5, 128, 2, 2, relu= False, name='dis_conv1', reuse=True)
         .lrelu(name='dis_conv1_lrelu')
         .conv(5, 5, 256, 2, 2, relu= False, name='dis_conv2', reuse=True)
         .bn(256, name='dis_conv2_bn')
         .lrelu(name='dis_conv2_lrelu')
         .conv(5, 5, 512, 2, 2, relu= False, name='dis_conv3', reuse=True)
         .bn(512, name='dis_conv3_bn')
         .lrelu(name='dis_conv3_lrelu')
         .conv(5, 5, 1024, 2, 2, relu= False, name='dis_conv4', reuse=True)
         .bn(1024, name='dis_conv4_bn')
         .lrelu(name='dis_conv4_lrelu')
         .conv(4, 4, 1, 4, 4, relu= False, name='dis_prob', reuse=True)
         .sigmoid(name='gen_dis'))


from tensorflow.examples.tutorials.mnist import input_data
import os

mnist = input_data.read_data_sets("../dataset/MNIST/", one_hot=True)

learning_rate = 0.001
training_iters = 10000
batch_size = 32
display_step = 10
d_steps = 3

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample, cmap='Greys_r')

    return fig


def train(net, sess):
    sess.run(init)
    if os.path.exists("../learn_param/mnist_gan.ckpt.meta"):
        net.load("../learn_param/mnist_gan.ckpt", sess, True)

    step = 1
    i = 0
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
        z_mb = sample_z(batch_size, z_dim)

        _, G_loss_curr = sess.run(
            [net.opt_gen, net.cost_gen],
            feed_dict={net.dataX: X_mb, net.dataZ: sample_z(batch_size, z_dim)}
        )

        if step % 1000 == 0:
            print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'
                  .format(step, D_loss_curr, G_loss_curr))
            #tmp_sample = sample_z(16, 64)

            samples = sess.run(net.get_output('gen_prob'), feed_dict={net.dataZ: sample_z(16, z_dim)})
            fig = plot(samples)
            plt.savefig('../test/test_{}.png'
                        .format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        step += 1
    save_path = net.saver.save(sess, "../learn_param/dcgan.ckpt")
    print("Model saved infile : {}".format(save_path))

if __name__ == '__main__':
    net = DCGAN(X_size=[28,28,3], Y_size=[z_dim], name='dcgan')

    gen_var = net.get_variables('gen_fc') + net.get_variables('gen_log_prob')
    dis_var = net.get_variables('dis_fc') + net.get_variables('dis_prob')

    net.set_cost_gen(0.5 * tf.reduce_mean((net.get_output('gen_dis') - 1)**2))
    net.set_optimizer_gen(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net.cost_gen, var_list= gen_var))

    net.set_cost_dis(0.5 * (tf.reduce_mean((net.get_output('real_dis') - 1)**2) + tf.reduce_mean(net.get_output('gen_dis')**2)))
    net.set_optimizer_dis(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net.cost_dis, var_list= dis_var))

    init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for i in range(10):
            train(net, sess)
            print("complete {} iteration".format(i+1))
