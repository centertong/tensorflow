from cnn.network import Network
import tensorflow as tf

n_classes = 1000

class Yolo(Network):
    def __init__(self, X_size, Y1_size, Y2_size, Y3_size, Y4_size, trainable=True, name=None):
        self.name = name
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None] + X_size)
        self.target1 = tf.placeholder(tf.float32, shape=[None] + Y1_size)
        self.target2 = tf.placeholder(tf.float32, shape=[None] + Y2_size)
        self.target3 = tf.placeholder(tf.float32, shape=[None] + Y3_size)
        self.target4 = tf.placeholder(tf.float32, shape=[None] + Y4_size)
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = {'data': self.data, 'target1': self.target1, 'target2': self.target2, 'target3': self.target3, 'target4': self.target4, 'keep_prob': self.keep_prob}
        self.trainable = trainable
        self.setup()
        self.saver = tf.train.Saver()

    def setup(self):
        (self.feed('data')
         .pad([[0, 0], [3, 3], [3, 3], [0, 0]], name='pad_1')
         .conv(7,7,64, 2, 2, padding='VALID', name='conv1')
         .max_pool(2,2,2,2, padding='SAME', name='pool1')
         .conv(3,3,192,1,1, name='conv2')
         .max_pool(2,2,2,2, padding='SAME', name='pool2')
         .conv(1,1,128,1,1, name='conv3')
         .conv(3,3,256,1,1, name='conv4')
         .conv(1,1,256,1,1, name='conv5')
         .conv(3,3,512,1,1, name='conv6')
         .max_pool(2,2,2,2, padding='SAME', name='pool3')
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
         .max_pool(2,2,2,2, padding='SAME', name='pool4')
         .conv(1,1,512,1,1, name='conv17')
         .conv(3,3,1024,1,1, name='conv18')
         .conv(1, 1, 512, 1, 1, name='conv19')
         .conv(3, 3, 1024, 1, 1, name='conv20')
         .conv(3,3,1024,1,1, name='conv21')
         .pad([[0, 0], [1, 1], [1, 1], [0, 0]], name='pad_2')
         .conv(3,3,1024,2,2, padding='VALID', name='conv22')
         .conv(3,3,1024,1,1, name='conv23')
         .conv(3,3,1024,1,1, name='conv24'))

        (self.feed('conv24')
         .fc(512, name='fc1')
         .fc(4096, name='fc11')
         .dropout(self.keep_prob, name='drop1'))

        (self.feed('drop1')
         .fc(7*7*20, name='fc2')
         .reshape([7, 7, 20], name='reshape_cls')
         .softmax(name='cls_prob'))

        (self.feed('drop1')
         .fc(7 * 7, relu=False, name='fc3')
         .reshape([7, 7], name='reshape_bbox_cls_1'))

        (self.feed('drop1')
         .fc(7 * 7 * 4, relu=False, name='fc4')
         .reshape([7, 7, 4], name='reshape_bbox_reg_1'))

        (self.feed('drop1')
         .fc(7 * 7, relu=False, name='fc5')
         .reshape([7, 7], name='reshape_bbox_cls_2'))

        (self.feed('drop1')
         .fc(7 * 7 * 4, relu=False, name='fc6')
         .reshape([7, 7, 4], name='reshape_bbox_reg_2'))


from cnn.factory import make_train_dataset
from sklearn.utils import shuffle

import os

image, target1, target2, target3, target4 = make_train_dataset(8000)

learning_rate = 0.01
training_iters = 5000
batch_size = 1
display_step = 1000

def train(net, sess):
    sess.run(init)
    #if os.path.exists("../learn_param/yolo.ckpt.meta"):
    #    net.load("../learn_param/yolo.ckpt", sess, True)

    step = 0
    while step * batch_size < training_iters:
        batch_xs = image[step * batch_size: (step + 1) * batch_size]
        batch_ys1 = target1[step * batch_size: (step + 1) * batch_size]
        batch_ys2 = target2[step * batch_size: (step + 1) * batch_size]
        batch_ys3 = target3[step * batch_size: (step + 1) * batch_size]
        batch_ys4 = target4[step * batch_size: (step + 1) * batch_size]

        feed_dict = {net.data: batch_xs, net.target1: batch_ys1, net.target2: batch_ys2, net.target3: batch_ys3, net.target4: batch_ys4, net.keep_prob: 0.4}

        sess.run(net.opt, feed_dict= feed_dict)

        if step % display_step == 0:
            feed_dict[net.keep_prob] = 1.
            #acc = sess.run(accuracy, feed_dict=feed_dict)
            loss = sess.run(net.cost, feed_dict=feed_dict)
            #loss1 = sess.run(net.target2, feed_dict=feed_dict)
            #loss2 = sess.run(net.get_output('reshape_bbox_reg_1'), feed_dict=feed_dict)
            print(loss)
            #print(loss1)
            #print(loss2)
        step += 1
    save_path = net.saver.save(sess, "../learn_param/yolo.ckpt")
    print("Model saved infile : {}".format(save_path))

if __name__ == '__main__':
    net = Yolo(X_size=[448,448,3], Y1_size=[7, 7], Y2_size=[7, 7, 4], Y3_size=[7, 7], Y4_size=[7, 7, 4], name='Yolo')
    #cost1 = tf.nn.l2_loss(net.get_output('reshape_bbox_cls_1') - net.target1) \
    cost1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net.get_output('reshape_bbox_cls_1'), labels=net.target1)) \
                    + tf.reduce_mean(tf.reduce_mean((net.get_output('reshape_bbox_reg_1') - net.target2)**2,axis=3) * net.target1)
    #cost2 = tf.nn.l2_loss(net.get_output('reshape_bbox_cls_2') - net.target3) \
    cost2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net.get_output('reshape_bbox_cls_2'), labels=net.target3)) \
            + tf.reduce_mean(tf.reduce_mean((net.get_output('reshape_bbox_reg_2') - net.target4)**2,axis=3) * net.target3)

    net.set_cost(cost1 + cost2)
    net.set_optimizer(tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(net.cost))

    #correct_pred = tf.equal(tf.argmax(net.get_output('cls_prob'), 1), tf.argmax(net.get_output('target'),1))
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    with tf.Session() as sess:
        for i in range(10):

            image, target1, target2, target3, target4 = shuffle(image, target1, target2, target3, target4)
            train(net, sess)
            print("complete {} iteration".format(i+1))