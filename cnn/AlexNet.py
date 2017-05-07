from cnn.network import Network
import tensorflow as tf

n_classes = 1000

class AlexNet(Network):
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
         .conv(11,11,96,4,4, name='conv1', padding='VALID') #padding = 0
         .max_pool(3,3,2,2, padding='VALID', name= 'pool1')
         .norm(name='norm1')
         .conv(5,5,256,1,1, name='conv2') #padding = 2
         .max_pool(3,3,2,2, padding='VALID', name='pool2')
         .norm(name='norm2')
         .conv(3,3,384,1,1, name='conv3') #padding = 1
         .conv(3,3,384,1,1, name='conv4') #padding = 1
         .conv(3,3,256,1,1, name='conv5') #padding = 1
         .max_pool(3,3,2,2, padding='VALID', name='pool3'))

        (self.feed('pool3')
         .fc(4096, name='fc1')
         .dropout(self.keep_prob, name='drop1')
         .fc(4096, name='fc2')
         .dropout(self.keep_prob, name='drop2')
         .fc(n_classes, relu=False, name='cls_score')
         .softmax(name='cls_prob'))

"""
SGD를 사용하고 mini-batch 사이즈는 128
모멘텀은 0.9, weight decay(L2)는 0.0005 (weight decay를 사용하니 트레이닝 에러가 줄었다고 한다.)
weight 초기화는 zero-mean gaussian * 0.01
dropout 확률은 0.5
learning rate는 0.01, validation error의 성능향상이 멈추면 10배 감소시킨다.
7개의 CNN을 앙상블하여 18.2% -> 15.4%로 향상시켰다.
"""


from tensorflow.examples.tutorials.mnist import input_data
import os

mnist = input_data.read_data_sets("../dataset/MNIST/", one_hot=True)

learning_rate = 0.01
training_iters = 10000
batch_size = 128
display_step = 10

def train(net, sess):
    sess.run(init)
    if os.path.exists("../learn_param/alexnet.ckpt.meta"):
        net.load("../learn_param/alexnet.ckpt", sess, True)

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
    save_path = net.saver.save(sess, "../learn_param/alexnet.ckpt")
    print("Model saved infile : {}".format(save_path))

if __name__ == '__main__':
    net = AlexNet(X_size=[227,227,3], Y_size=[n_classes], name='AlexNet')
    net.set_cost(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net.get_output('cls_score'), labels=net.target)))
    net.set_optimizer(tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(net.cost))

    correct_pred = tf.equal(tf.argmax(net.get_output('cls_prob'), 1), tf.argmax(net.get_output('target'),1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for i in range(10):
            train(net, sess)
            print("complete {} iteration".format(i+1))