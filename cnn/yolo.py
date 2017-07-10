from cnn.network import Network
import tensorflow as tf

n_classes = 1000

class Yolo(Network):
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
         .max_pool(2,2,2,2, name='pool1')
         .conv(3,3,192,1,1, name='conv2')
         .max_pool(2,2,2,2, name='pool2')
         .conv(1,1,128,1,1, name='conv3')
         .conv(3,3,256,1,1, name='conv4')
         .conv(1,1,256,1,1, name='conv5')
         .conv(3,3,512,1,1, name='conv6')
         .max_pool(2,2,2,2, name='pool3')
         .conv(1,1,256,1,1, name='conv7')
         .conv(3,3,512,1,1, name='conv8')
         .conv(1, 1, 256, 1, 1, name='conv9')
         .conv(3, 3, 512, 1, 1, name='conv10')
         .conv(1, 1, 256, 1, 1, name='conv11')
         .conv(3, 3, 512, 1, 1, name='conv12')
         .conv(1, 1, 256, 1, 1, name='conv13')
         .conv(3, 3, 512, 1, 1, name='conv14')
         .conv(1,1,512,1,1, name='conv15')
         .conv(3,3,1024,1,1, name='conv16')
         .max_pool(2,2,2,2, name='pool4')
         .conv(1,1,512,1,1, name='conv17')
         .conv(3,3,1024,1,1, name='conv18')
         .conv(1, 1, 512, 1, 1, name='conv19')
         .conv(3, 3, 1024, 1, 1, name='conv20')
         .conv(3,3,1024,1,1, name='conv21')
         .conv(3,3,1024,2,2, name='conv22')
         .conv(3,3,1024,1,1, name='conv23')
         .conv(3,3,1024,1,1, name='conv24'))

        (self.feed('conv24')
         .fc(4096, name='fc1'))

        (self.feed('fc1')
         .fc(7*7*20, name='fc2')
         .reshape([7,7,20], name='reshape_cls')
         .softmax(name='cls_prob'))

        (self.feed('fc1')
         .fc(4096, name='fc3')
         .reshape([7,7,10], name='reshape_bbox'))

from cnn.factory import make_train_dataset
import os

image, target = make_train_dataset()

learning_rate = 0.01
training_iters = 10000
batch_size = 10
display_step = 100

def train(net, sess):
    sess.run(init)
    if os.path.exists("../learn_param/resnet.ckpt.meta"):
        net.load("../learn_param/resnet.ckpt", sess, True)

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
    save_path = net.saver.save(sess, "../learn_param/resnet.ckpt")
    print("Model saved infile : {}".format(save_path))

if __name__ == '__main__':
    net = Yolo(X_size=[400,400,3], Y_size=[n_classes], name='Yolo')
    net.set_cost(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net.get_output('cls_score'), labels=net.target)))
    net.set_optimizer(tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(net.cost))

    correct_pred = tf.equal(tf.argmax(net.get_output('cls_prob'), 1), tf.argmax(net.get_output('target'),1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for i in range(10):
            train(net, sess)
            print("complete {} iteration".format(i+1))