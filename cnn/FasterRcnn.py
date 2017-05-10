from cnn.network import Network
import tensorflow as tf

n_classes = 1000

class RCNN(Network):
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
         .max_pool(3,3,2,2, name='pool1'))

        (self.residual_v2('pool1', 64, 64, 256, 1, name='res1_1'))
        (self.residual_v2('res1_1', 64, 64, 256, 1, name='res1_2'))
        (self.residual_v2('res1_2', 64, 64, 256, 1, name='res1_3'))

        (self.residual_v2('res1_3', 128, 128, 512, 2, name='res2_1'))
        (self.residual_v2('res2_1', 128, 128, 512, 1, name='res2_2'))
        (self.residual_v2('res2_2', 128, 128, 512, 1, name='res2_3'))
        (self.residual_v2('res2_3', 128, 128, 512, 1, name='res2_4'))

        (self.residual_v2('res2_4', 256, 256, 1024, 2, name='res3_1'))
        (self.residual_v2('res3_1', 256, 256, 1024, 1, name='res3_2'))
        (self.residual_v2('res3_2', 256, 256, 1024, 1, name='res3_3'))
        (self.residual_v2('res3_3', 256, 256, 1024, 1, name='res3_4'))
        (self.residual_v2('res3_4', 256, 256, 1024, 1, name='res3_5'))
        (self.residual_v2('res3_5', 256, 256, 1024, 1, name='res3_6'))

        (self.residual_v2('res3_6', 512, 512, 2048, 2, name='res4_1'))
        (self.residual_v2('res4_1', 512, 512, 2048, 1, name='res4_2'))
        (self.residual_v2('res4_2', 512, 512, 2048, 1, name='res4_3'))

        (self.feed('res4_3')
         .avg_pool(7, 7, 1,1, name='pool2')
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
    net = RCNN(X_size=[227,227,3], Y_size=[n_classes], name='RCNN')
    net.set_cost(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net.get_output('cls_score'), labels=net.target)))
    net.set_optimizer(tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(net.cost))

    correct_pred = tf.equal(tf.argmax(net.get_output('cls_prob'), 1), tf.argmax(net.get_output('target'),1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for i in range(10):
            train(net, sess)
            print("complete {} iteration".format(i+1))