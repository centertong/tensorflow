import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import scipy.ndimage.interpolation

from gan.network import Network

mb_size = 32
X_dim = 784
h_dim = 128
learning_rate = 1e-3
training_iters = 10000
d_steps = 3

class DiscoGAN(Network):
    def __init__(self, X_size, Y_size , trainable=True, name=None):
        self.name = name
        self.inputs = []
        self.dataA = tf.placeholder(tf.float32, shape = [None] + X_size)
        self.dataB = tf.placeholder(tf.float32, shape = [None] + Y_size)
        self.layers = {'dataA':self.dataA, 'dataB':self.dataB}
        self.trainable = trainable
        self.setup()
        self.saver = tf.train.Saver()

    def setup(self):
        #generator B -> A
        (self.feed('dataB')
         .fc(h_dim, name='gen_BA_1')
         .fc(784, name='gen_BA_2', relu=False)
         .sigmoid(name='gen_BA'))

        #Discriminator A from Gen BA
        (self.feed('gen_BA')
         .fc(h_dim, name='dis_A_1')
         .fc(1, name='dis_A_2', relu=False)
         .sigmoid(name='dis_A_fake'))

        # Discriminator A from DATA A
        (self.feed('dataA')
         .fc(h_dim, name='dis_A_1', reuse=True)
         .fc(1, name='dis_A_2', relu=False, reuse=True)
         .sigmoid(name='dis_A_real'))

        # generator A -> B
        (self.feed('dataA')
         .fc(h_dim, name='gen_AB_1')
         .fc(784, name='gen_AB_2', relu=False)
         .sigmoid(name='gen_AB'))

        #Discriminator B from Gen AB
        (self.feed('gen_AB')
         .fc(h_dim, name='dis_B_1')
         .fc(1, name='dis_B_2', relu=False)
         .sigmoid(name='dis_B_fake'))

        # Discriminator B from DATA B
        (self.feed('dataB')
         .fc(h_dim, name='dis_B_1', reuse=True)
         .fc(1, name='dis_B_2', relu=False, reuse=True)
         .sigmoid(name='dis_B_real'))

        #Generator B -> A from gen AB
        (self.feed('gen_AB')
         .fc(h_dim, name='gen_BA_1', reuse=True)
         .fc(784, name='gen_BA_2', relu=False, reuse=True)
         .sigmoid(name='gen_ABA'))

        #Generator B -> A from gen AB
        (self.feed('gen_BA')
         .fc(h_dim, name='gen_AB_1', reuse=True)
         .fc(784, name='gen_AB_2', relu=False, reuse=True)
         .sigmoid(name='gen_BAB'))


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

def sample_X(X, size):
    start_idx = np.random.randint(0, X.shape[0]-size)
    return X[start_idx:start_idx+size]

mnist = input_data.read_data_sets("../dataset/MNIST/", one_hot=True)

def train(net, sess):
    sess.run(init)
    if os.path.exists("../learn_param/disco_mnist.ckpt.meta"):
        net.load("../learn_param/disco_mnist.ckpt", sess, True)

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
    net = DiscoGAN(X_size=[784], Y_size=[784], name='disco_mnist')

    gen_var = net.get_variables('gen_fc') + net.get_variables('gen_log_prob')
    dis_var = net.get_variables('dis_fc') + net.get_variables('dis_prob')


    # GAN cost function 정의 시 주의사항 : network에서 마지막 layer에 Sigmoid 유무를 잘 확인하자.
    # 1,2 는 둘다 Sigmoid가 필요
    # 3은 discriminator에서는 불필요 (왜냐하면 cost 함수에 sigmoid를 포함하고 있기 때문이다.)

    """
    net.set_cost_gen(0.5 * tf.reduce_mean((net.get_output('gen_dis') - 1)**2))
    net.set_optimizer_gen(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net.cost_gen, var_list= gen_var))
    net.set_cost_dis(0.5 * (tf.reduce_mean((net.get_output('real_dis') - 1)**2) + tf.reduce_mean(net.get_output('gen_dis')**2)))
    net.set_optimizer_dis(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net.cost_dis, var_list= dis_var))
    """

    net.set_cost_gen(tf.reduce_mean(tf.log(net.get_output('gen_dis'))))
    net.set_optimizer_gen(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-net.cost_gen, var_list=gen_var))
    net.set_cost_dis(tf.reduce_mean(tf.log(net.get_output('real_dis')) + tf.log(1 - net.get_output('gen_dis'))))
    net.set_optimizer_dis(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-net.cost_dis, var_list=dis_var))

    """
    net.set_cost_gen(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net.get_output('gen_dis'), labels=tf.ones_like(net.get_output('gen_dis')))))
    net.set_optimizer_gen(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net.cost_gen, var_list=gen_var))

    dis_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net.get_output('real_dis'), labels=tf.ones_like(net.get_output('real_dis')))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net.get_output('gen_dis'),labels=tf.zeros_like(net.get_output('gen_dis'))))
    net.set_cost_dis(dis_cost)
    net.set_optimizer_dis(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net.cost_dis, var_list=dis_var))
    """


    init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for i in range(10):
            train(net, sess)
            print("complete {} iteration".format(i+1))
