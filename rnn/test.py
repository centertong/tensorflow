# 챗봇, 번역, 이미지 캡셔닝등에 사용되는 시퀀스 학습/생성 모델인 Seq2Seq 을 구현해봅니다.
import tensorflow as tf
import numpy as np

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
learning_rate = 0.01
# 입력과 출력의 형태가 one-hot 인코딩으로 같으므로 크기도 같다.
n_class = n_input = dic_len
n_hidden = 128


#########
# 신경망 모델 구성
######
# Seq2Seq 모델은 인코더의 입력과 디코더의 입력의 형식이 같다.
# [batch size, time steps, input size]
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
# [batch size, time steps]
targets = tf.placeholder(tf.int64, [None, None])

W = tf.Variable(tf.ones([n_hidden, n_class]))
b = tf.Variable(tf.zeros([n_class]))

# tf.nn.dynamic_rnn 옵션에서 time_major 값을 True 로 설정
# [batch_size, n_steps, n_input] -> Tensor[n_steps, batch_size, n_input]
enc_input = tf.transpose(enc_input, [1, 0, 2])
dec_input = tf.transpose(dec_input, [1, 0, 2])

# 인코더 셀을 구성한다.
with tf.variable_scope('encode'):
    enc_cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
    enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, output_keep_prob=0.5)

    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
                                            dtype=tf.float32)

# 디코더 셀을 구성한다.
with tf.variable_scope('decode'):
    dec_cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    # Seq2Seq 모델은 인코더 셀의 최종 상태값을
    # 디코더 셀의 초기 상태값으로 넣어주는 것이 핵심.
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                            initial_state=enc_states,
                                            dtype=tf.float32)


# sparse_softmax_cross_entropy_with_logits 함수를 사용하기 위해
# 각각의 텐서의 차원들을 다음과 같이 변형하여 계산한다.
#    -> [batch size, time steps, hidden layers]
time_steps = tf.shape(outputs)[1]
#    -> [batch size * time steps, hidden layers]
outputs_trans = tf.reshape(outputs, [-1, n_hidden])
#    -> [batch size * time steps, class numbers]
model = tf.matmul(outputs_trans, W) + b
#    -> [batch size, time steps, class numbers]
model = tf.reshape(model, [-1, time_steps, n_class])


cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model, labels=targets))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


#########
# 신경망 모델 학습
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, output_batch, target_batch = make_batch(seq_data)

for epoch in range(100):
    _, loss = sess.run([optimizer, cost],
                       feed_dict={enc_input: input_batch,
                                  dec_input: output_batch,
                                  targets: target_batch})

    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')


#########
# 입력만으로 다음 시퀀스를 예측해보자
######
# 시퀀스 데이터를 받아 다음 결과를 예측하고 디코딩하는 함수
def decode(seq_data):
    prediction = tf.argmax(model, 2)

    input_batch, output_batch, target_batch = make_batch([seq_data])

    result = sess.run(prediction,
                      feed_dict={enc_input: input_batch,
                                 dec_input: output_batch,
                                 targets: target_batch})

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


print('\n=== 전체 시퀀스를 한 번에 예측 ===')

# 결과를 모르므로 빈 시퀀스 값인 P로 값을 채웁니다.
seq_data = ['word', 'PPPP']
print('word ->', decode_at_once(seq_data))

seq_data = ['wodr', 'PPPP']
print('wodr ->', decode_at_once(seq_data))

seq_data = ['love', 'PPPP']
print('love ->', decode_at_once(seq_data))

seq_data = ['loev', 'PPPP']
print('loev ->', decode_at_once(seq_data))

seq_data = ['abcd', 'PPPP']
print('abcd ->', decode_at_once(seq_data))


print('\n=== 한글자씩 점진적으로 시퀀스를 예측 ===')

seq_data = ['word', '']
print('word ->', decode_step_by_step(seq_data))

seq_data = ['wodr', '']
print('wodr ->', decode_step_by_step(seq_data))

seq_data = ['love', '']
print('love ->', decode_step_by_step(seq_data))

seq_data = ['loev', '']
print('loev ->', decode_step_by_step(seq_data))

seq_data = ['abcd', '']
print('abcd ->', decode_step_by_step(seq_data))