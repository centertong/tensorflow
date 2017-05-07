from cnn.network import Network
import tensorflow as tf

n_classes = 1000

class GoogleNet(Network):
    def __init__(self, X_size, Y_size , trainable=True, name=None):
        self.name = name
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape = [None] + X_size)
        self.target = tf.placeholder(tf.float32, shape = [None] + Y_size)
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = {'data':self.data, 'target':self.target, 'keep_prob':self.keep_prob}
        self.trainable = trainable
        self.setup()
        self.saver = tf.train.Saver()

    def setup(self):
        (self.feed('data')
         .conv(7,7,64, 2, 2, name='conv1')
         .max_pool(3,3,2,2, name='pool1')
         .conv(3,3,64,1,1, name='conv2')
         .conv(3,3,192,1,1, name='conv3')
         .max_pool(3,3,2,2, name='pool2'))

        (self.inception('pool2', 64, 96, 128, 16, 32, 32, name='inception3a'))
        (self.inception('inception3a', 128, 128, 192, 32, 96, 64, name='inception3b'))
        (self.feed('inception3b').max_pool(3,3,2,2,name='pool3'))

        (self.inception('pool3', 192, 96, 208, 16, 48, 64, name='inception4a'))
        (self.inception('inception4a', 160, 112, 224, 24, 64, 64, name='inception4b'))
        (self.inception('inception4b', 128, 128, 256, 24, 64, 64, name='inception4c'))
        (self.inception('inception4c', 112, 144, 288, 32, 64, 64, name='inception4d'))
        (self.inception('inception4d', 256, 160, 320, 32, 128, 128, name='inception4e'))
        (self.feed('inception4e').max_pool(3,3,2,2,name='pool4'))

        (self.inception('pool4', 256, 160, 320, 32, 128, 128, name='inception5a'))
        (self.inception('inception5a', 384, 192, 384, 48, 128, 128, name='inception5b'))

        (self.feed('inception5b')
         .avg_pool(7,7,1,1,name='pool5')
         .dropout(self.keep_prob, name='drop')
         .fc(n_classes, name='cls_score')
         .softmax(name='cls_prob'))

from tensorflow.examples.tutorials.mnist import input_data
import os

mnist = input_data.read_data_sets("../dataset/MNIST/", one_hot=True)

learning_rate = 0.01
training_iters = 10000
batch_size = 128
display_step = 10

def train(net, sess):
    sess.run(init)
    if os.path.exists("../learn_param/googlenet.ckpt.meta"):
        net.load("../learn_param/googlenet.ckpt", sess, True)

    step = 1
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28,28 , 1)
        feed_dict = {net.data: batch_xs, net.target: batch_ys, net.keep_prob: 0.4}

        sess.run(net.opt, feed_dict= feed_dict)

        if step % display_step == 0:
            feed_dict[net.keep_prob] = 1.
            acc = sess.run(accuracy, feed_dict=feed_dict)
            loss = sess.run(net.cost, feed_dict=feed_dict)
            print(acc, loss)
        step += 1
    save_path = net.saver.save(sess, "../learn_param/googlenet.ckpt")
    print("Model saved infile : {}".format(save_path))

if __name__ == '__main__':
    net = GoogleNet(X_size=[227,227,3], Y_size=[n_classes], name='GoogleNet')
    net.set_cost(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net.get_output('cls_score'), labels=net.target)))
    net.set_optimizer(tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(net.cost))

    correct_pred = tf.equal(tf.argmax(net.get_output('cls_prob'), 1), tf.argmax(net.get_output('target'),1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for i in range(10):
            train(net, sess)
            print("complete {} iteration".format(i+1))