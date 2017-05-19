# -*- coding: utf-8 -*-
# 자연어 처리나 음성 처리 분야에 많이 사용되는 RNN 의 기본적인 사용법을 익힙니다.
# 1 부터 0 까지 순서대로 숫자를 예측하여 세는 모델을 만들어봅니다.

import tensorflow as tf
import numpy as np
from rnn.network import Network

import os


num_arr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
# one-hot 인코딩을 사용하기 위해 연관 배열을 만듭니다.
# {'1': 0, '2': 1, '3': 2, ..., '9': 9, '0', 10}
num_dic = {n: i for i, n in enumerate(num_arr)}
dic_len = len(num_dic)

# 다음 배열은 입력값과 출력값으로 다음처럼 사용할 것 입니다.
# 123 -> X, 4 -> Y
# 234 -> X, 5 -> Y
seq_data = ['1234', '2345', '3456', '4567', '5678', '6789', '7890']


# 위의 데이터에서 X,Y 값을 뽑아 one-hot 인코딩을 한 뒤 배치데이터로 만드는 함수
def one_hot_seq(seq_data):
    x_batch = []
    y_batch = []
    for seq in seq_data:
        # 여기서 생성하는 x_data 와 y_data 는
        # 실제 숫자가 아니라 숫자 리스트의 인덱스 번호 입니다.
        # [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5] ...
        x_data = [num_dic[n] for n in seq[:-1]]
        # 3, 4, 5, 6...10
        y_data = num_dic[seq[-1]]
        # one-hot 인코딩을 합니다.
        # if x_data is [0, 1, 2]:
        # [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]]
        x_batch.append(np.eye(dic_len)[x_data])
        # if 3: [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]
        y_batch.append(np.eye(dic_len)[y_data])

    return x_batch, y_batch


#########
# 옵션 설정
######
# 입력값 크기. 10개의 숫자에 대한 one-hot 인코딩이므로 10개가 됩니다.
# 예) 3 => [0 0 1 0 0 0 0 0 0 0 0]
n_input = 10
# 타입 스텝: [1 2 3] => 3
# RNN 을 구성하는 시퀀스의 갯수입니다.
n_steps = 3
# 출력값도 입력값과 마찬가지로 10개의 숫자로 분류합니다.
n_classes = 10
# 히든 레이어의 특성치 갯수
n_hidden = 128
# 학습 비율
learning_rate = 0.01
# 학습 횟수
n_iteration = 100
# 중간 결과값 출력
display_step = 10

#########
# 신경망 모델 구성
######

class counting_RNN(Network):
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
        (self.feed('data')
         .rnn(n_hidden, n_layer= 1, name='rnn')
         .fc(n_classes, name='fc'))


#########
# 신경망 모델 학습
######
def train(net, sess):
    sess.run(init)
    if os.path.exists("../learn_param/counting_rnn.ckpt.meta"):
        net.load("../learn_param/counting_rnn.ckpt", sess, True)

    step = 1

    x_batch, y_batch = one_hot_seq(seq_data)

    while step < n_iteration:
        feed_dict = {net.data: x_batch, net.target: y_batch}

        sess.run(net.opt, feed_dict= feed_dict)

        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict=feed_dict)
            loss = sess.run(net.cost, feed_dict=feed_dict)
            print(acc, loss)
        step += 1
    save_path = net.saver.save(sess, "../learn_param/counting_rnn.ckpt")
    print("Model saved infile : {}".format(save_path))


if __name__ == '__main__':
    net = counting_RNN(X_size=[n_steps, n_input], Y_size=[n_classes], name='counting')
    net.set_cost(tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=net.get_output('fc'), labels=net.target)))
    net.set_optimizer(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net.cost))

    prediction = tf.argmax(net.get_output('fc'), 1)
    prediction_check = tf.equal(prediction, tf.argmax(net.target, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

    init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for i in range(10):
            train(net, sess)
            print("complete {} iteration".format(i + 1))