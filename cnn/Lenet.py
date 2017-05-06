from cnn.network import Network
import tensorflow as tf

n_classes = 10

class LeNet(Network):
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
         .conv(5,5,32,1,1, name='conv1')
         .max_pool(2,2,2,2, padding='VALID', name= 'pool1')
         .conv(5,5,64,1,1, name='conv2'))

        (self.feed('conv2')
         .fc(12544, name='fc1')
         .dropout(self.keep_prob, name='drop1')
         .fc(n_classes, relu=False, name='cls_score')
         .softmax(name='cls_prob'))

from tensorflow.examples.tutorials.mnist import input_data
import os

mnist = input_data.read_data_sets("../dataset/MNIST/", one_hot=True)

learning_rate = 0.001
training_iters = 10000
batch_size = 100
display_step = 10

def train(net, sess):
    sess.run(init)
    if os.path.exists("../learn_param/lenet.ckpt.meta"):
        net.load("../learn_param/lenet.ckpt", sess, True)

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
    save_path = net.saver.save(sess, "../learn_param/lenet.ckpt")
    print("Model saved infile : {}".format(save_path))

if __name__ == '__main__':
    net = LeNet(X_size=[28,28,1], Y_size=[10], name='LeNet')
    net.set_cost(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net.get_output('cls_score'), labels=net.target)))
    net.set_optimizer(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net.cost))

    correct_pred = tf.equal(tf.argmax(net.get_output('cls_prob'), 1), tf.argmax(net.get_output('target'),1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for i in range(10):
            train(net, sess)
            print("complete {} iteration".format(i+1))