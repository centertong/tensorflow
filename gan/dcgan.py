import tensorflow as tf
import numpy as np
from gan.network import Network
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec

import os
import cv2

z_dim = 200

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
         .reshape([-1,1,1,200], name='gen_reshape')
         .deconv(1,1, 1024,4,4, name='gen_deconv1', padding='VALID')
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
         .conv(4,4, 1,1, 1, relu= False, name='dis_prob', padding='VALID')
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
         .conv(4, 4, 1, 1,1, relu= False, name='dis_prob', reuse=True, padding='VALID')
         .sigmoid(name='gen_dis'))



def read_dataset(path):
    file_list = os.listdir(path)
    dataset = []
    for file in file_list:
        if file.split('.')[-1] == 'png':
            img = cv2.imread(os.path.join(path, file))
            w,h,c = img.shape
            wp = 0
            hp = 0
            if w % 64:
                wp = 1
            if h % 64:
                hp = 1

            for i in range(w // 64):
                for j in range(h // 64):
                    tmp = img[i * (64 + wp): i * (64 + wp) + 64, j * (64 + hp): j * (64 + hp) + 64]
                    dataset.append(tmp)
    return np.array(dataset)


dataset = read_dataset('../dataset/pokemon/GenI')

learning_rate = 0.001
training_iters = 500
batch_size = 10
display_step = 10
d_steps = 1

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

"""
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
"""


def train(net, sess):
    sess.run(init)
    if os.path.exists("../learn_param/dcgan2.ckpt.meta"):
        net.load("../learn_param/dcgan2.ckpt", sess, True)

    step = 0
    i = 0
    while step < training_iters:
        D_loss_curr = None
        for _ in range(d_steps):
            X_mb = dataset[step * batch_size: (step+1) * batch_size]
            z_mb = sample_z(batch_size, z_dim)

            _, D_loss_curr = sess.run(
                [net.opt_dis, net.cost_dis],
                feed_dict={net.dataX: X_mb, net.dataZ: z_mb}
            )

        X_mb = dataset[step * batch_size: (step+1) * batch_size]
        z_mb = sample_z(batch_size, z_dim)

        _, G_loss_curr = sess.run(
            [net.opt_gen, net.cost_gen],
            feed_dict={net.dataX: X_mb, net.dataZ: sample_z(batch_size, z_dim)}
        )

        step += 1
        if step % 1000 == 0:
            print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'
                  .format(step, D_loss_curr, G_loss_curr))
            #tmp_sample = sample_z(16, 64)

            samples = sess.run(net.get_output('gen_prob'), feed_dict={net.dataZ: sample_z(16, z_dim)})
            #fig = plot(samples)
            #plt.savefig('../test/test_{}.png'
            #            .format(str(i).zfill(3)), bbox_inches='tight')
            #i += 1
            #plt.close(fig)


    save_path = net.saver.save(sess, "../learn_param/dcgan.ckpt")
    print("Model saved infile : {}".format(save_path))

if __name__ == '__main__':
    net = DCGAN(X_size=[64,64,3], Y_size=[z_dim], batch_size=batch_size, name='dcgan')

    gen_var = net.get_variables('gen_deconv1') + net.get_variables('gen_deconv2') + net.get_variables('gen_deconv3') + net.get_variables('gen_deconv4') + net.get_variables('gen_deconv5')
    dis_var = net.get_variables('dis_prob') + net.get_variables('dis_conv1') + net.get_variables('dis_conv2') + net.get_variables('dis_conv3') + net.get_variables('dis_conv4')

    net.set_cost_gen(0.5 * tf.reduce_mean((net.get_output('gen_dis') - 1)**2))
    net.set_optimizer_gen(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net.cost_gen, var_list= gen_var))

    net.set_cost_dis(0.5 * (tf.reduce_mean((net.get_output('real_dis') - 1)**2) + tf.reduce_mean(net.get_output('gen_dis')**2)))
    net.set_optimizer_dis(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net.cost_dis, var_list= dis_var))

    init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for i in range(10):
            train(net, sess)
            print("complete {} iteration".format(i+1))
