# -*- coding: utf-8 -*-
# 챗봇, 번역, 이미지 캡셔닝등에 사용되는 시퀀스 학습/생성 모델인 Seq2Seq 을 구현해봅니다.

import tensorflow as tf
import numpy as np
from rnn.network import Network
import os

# S: 디코딩 입력의 시작을 나타내는 심볼
# E: 디코딩 출력을 끝을 나타내는 심볼
# P: 현재 배치 데이터의 time step 크기보다 작은 경우 빈 시퀀스를 채우는 심볼
#    예) 현재 배치 데이터의 최대 크기가 4 인 경우
#       word -> ['w', 'o', 'r', 'd']
#       to   -> ['t', 'o', 'P', 'P']
char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 영어를 한글로 번역하기 위한 학습 데이터
seq_data = [['word', '단어'], ['wood', '나무'],
            ['game', '놀이'], ['girl', '소녀'],
            ['kiss', '키스'], ['love', '사랑']]


def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        # 입력값과 출력값의 time step 을 같게 하기 위해 P 를 앞에 붙여준다. (안해도 됨)
        input = [num_dic[n] for n in ('P' + seq[0])]
        # 디코더 셀의 입력값. 시작을 나타내는 S 심볼을 맨 앞에 붙여준다.
        output = [num_dic[n] for n in ('S' + seq[1])]
        # 학습을 위해 비교할 출력값. 끝나는 것을 알려주기 위해 마지막에 E 를 붙인다.
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        # 출력값만 one-hot 인코딩이 아님 (sparse_softmax_cross_entropy_with_logits 사용)
        target_batch.append(target)

    return input_batch, output_batch, target_batch

#########
# 옵션 설정
######
# 입력과 출력의 형태가 one-hot 인코딩으로 같으므로 크기도 같다.
n_classes = n_input = dic_len
n_hidden = 128
n_layers = 3
n_iteration = 100
display_step = 10
learning_rate = 0.01

#########
# 신경망 모델 구성
######
# Seq2Seq 모델은 인코더의 입력과 디코더의 입력의 형식이 같다.

class Seq2Seq(Network):
    def __init__(self, X_size, Y_size, Z_size , trainable=True, name=None):
        self.name = name
        self.inputs = []
        self.data_inc = tf.placeholder(tf.float32, shape = [None] + X_size)
        self.data_dec = tf.placeholder(tf.float32, shape = [None] + Y_size)
        self.target = tf.placeholder(tf.int64, shape = [None] + Z_size)
        self.layers = {'data_inc':self.data_inc, 'data_dec':self.data_dec, 'target':self.target}
        self.trainable = trainable
        self.setup()
        self.saver = tf.train.Saver()

    def setup(self):
        (self.feed('data_inc')
         .rnn(n_hidden, n_layer=1, name='incoder'))

        (self.feed('data_dec')
         .rnn(n_hidden, n_layer=1, name='decoder', initial_state=self.get_output('incoder_s')))

        (self.feed('decoder')
         .shape(name='decoder_shape'))

        time_steps = self.get_output('decoder_shape')[1]

        (self.feed('decoder')
         .reshape([-1, n_hidden], name='decoder_reshape')
         .fc(n_classes, name='seq_reshape')
         .reshape([-1, time_steps, n_classes], name='seq'))

#########
# 신경망 모델 학습
######
def train(net, sess):
    sess.run(init)
    if os.path.exists("../learn_param/seq2seq.ckpt.meta"):
        net.load("../learn_param/seq2seq.ckpt", sess, True)

    step = 1

    x_batch, y_batch, target_batch = make_batch(seq_data)

    while step < n_iteration:
        feed_dict = {net.data_inc: x_batch, net.data_dec: y_batch, net.target: target_batch}

        sess.run(net.opt, feed_dict= feed_dict)

        if step % display_step == 0:
            loss , result= sess.run([net.cost, prediction], feed_dict=feed_dict)
            print(loss)

        step += 1
    save_path = net.saver.save(sess, "../learn_param/seq2seq.ckpt")
    print("Model saved infile : {}".format(save_path))






if __name__ == '__main__':
    net = Seq2Seq(X_size=[None, n_input], Y_size=[None, n_input], Z_size=[None], name='seq2seq')

    net.set_cost(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net.get_output('seq'), labels=net.target)))
    net.set_optimizer(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net.cost))

    prediction = tf.argmax(net.get_output('seq'), 2)

    init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for i in range(10):
            train(net, sess)
            print("complete {} iteration".format(i + 1))



        #########
        # 공부!!
        ######
        # 시퀀스 데이터를 받아 다음 결과를 예측하고 디코딩하는 함수
        def decode(seq_data):
            prediction = tf.argmax(net.get_output('seq'), 2)

            input_batch, output_batch, target_batch = make_batch([seq_data])

            result = sess.run(prediction,
                              feed_dict={net.data_inc: input_batch,
                                         net.data_dec: output_batch,
                                         net.target: target_batch})

            decode_seq = [[char_arr[i] for i in dec] for dec in result][0]

            return decode_seq


        # 한 번에 전체를 예측하고 E 이후의 글자를 잘라 단어를 완성
        def decode_at_once(seq_data):
            seq = decode(seq_data)
            end = seq.index('E')
            seq = ''.join(seq[:end])

            return seq


        # 시퀀스 데이터를 받아 다음 한글자를 예측하고,
        # 종료 심볼인 E 가 나올때까지 점진적으로 예측하여 최종 결과를 만드는 함수
        def decode_step_by_step(seq_data):
            decode_seq = ''
            current_seq = ''

            while current_seq != 'E':
                decode_seq = decode(seq_data)
                seq_data = [seq_data[0], ''.join(decode_seq)]
                current_seq = decode_seq[-1]

            return decode_seq

        # 결과를 모르므로 빈 시퀀스 값인 P로 값을 채웁니다.
        seq_data2 = ['word', 'PPPP']
        print('word ->', decode_at_once(seq_data2))


        seq_data3 = ['word', '']
        print('word ->', decode_step_by_step(seq_data3))
