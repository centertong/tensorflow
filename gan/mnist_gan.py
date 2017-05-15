import tensorflow as tf
import numpy as np
from gan.network import Network
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class mnist_gan(Network):
    def __init__(self, X_size, Y_size , trainable=True, name=None):
        self.name = name
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape = [None] + X_size)
        self.target = tf.placeholder(tf.float32, shape = [None] + Y_size)
        self.layers = {'data':self.data, 'target':self.target}
        self.trainable = trainable
        self.setup()
        self.saver = tf.train.Saver()

    def setup(self):
        #generator
        (self.feed('target')
         .fc(128, name='gen_fc')
         .fc(784, name='gen_log_prob', relu=False)
         .sigmoid(name='gen_prob'))

        #discriminator
        (self.feed('data')
         .fc(128, name='dis_fc')
         .fc(1, name='dis_prob', relu=False)
         .rename(name='real_dis'))

        (self.feed('gen_prob')
         .fc(128, name='dis_fc', reuse=True)
         .fc(1, name='dis_prob', relu=False, reuse=True)
         .rename(name='gen_dis'))


from tensorflow.examples.tutorials.mnist import input_data
import os

mnist = input_data.read_data_sets("../dataset/MNIST/", one_hot=True)

learning_rate = 0.001
training_iters = 100000
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
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

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
            z_mb = sample_z(batch_size, 64)

            _, D_loss_curr = sess.run(
                [net.opt_dis, net.cost_dis],
                feed_dict={net.data: X_mb, net.target: z_mb}
            )

        X_mb, _ = mnist.train.next_batch(batch_size)
        z_mb = sample_z(batch_size, 64)

        _, G_loss_curr = sess.run(
            [net.opt_gen, net.cost_gen],
            feed_dict={net.data: X_mb, net.target: sample_z(batch_size, 64)}
        )

        if step % 1000 == 0:
            print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'
                  .format(step, D_loss_curr, G_loss_curr))
            #tmp_sample = sample_z(16, 64)

            samples = sess.run(net.get_output('gen_prob'), feed_dict={net.target: sample_z(16, 64)})
            fig = plot(samples)
            plt.savefig('../test/test_{}.png'
                        .format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        step += 1
    save_path = net.saver.save(sess, "../learn_param/mnist_gan.ckpt")
    print("Model saved infile : {}".format(save_path))

if __name__ == '__main__':
    net = mnist_gan(X_size=[784], Y_size=[64], name='mnist')

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
