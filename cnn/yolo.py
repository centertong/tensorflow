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

