import numpy as np
import tensorflow as tf

DEFAULT_PADDING = 'SAME'  # 'SAME' or 'VALID'


def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


def rnn_layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output, layer_state = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        self.layers[name+'_s'] = layer_state
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated


class Network(object):
    def __init__(self, inputs, trainable=True, name=None):
        self.name = name
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()
        self.saver = tf.train.Saver()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False):
        if data_path.endswith('.ckpt'):
            self.saver.restore(session, data_path)
        else:
            data_dict = np.load(data_path).item()
            for key in data_dict:
                with tf.variable_scope(key, reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print("assign pretrain model " + subkey + " to " + key)
                        except ValueError:
                            print("ignore " + key)
                            if not ignore_missing:
                                raise

    def feed(self, *args):
        assert len(args) != 0
        self.inputs = []
        for layer in args:
            if isinstance(layer, str):
                try:
                    layer = self.layers[layer]
                    print(layer)
                except KeyError:
                    print(self.layers.keys())
                    raise KeyError('Unknown layer name fed: %s' % layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print(self.layers.keys())
            raise KeyError('Unknown layer name fed: %s' % layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    def get_variables(self, name):
        with tf.variable_scope(self.name + name, reuse=True) as scope:
            kernel = tf.get_variable('weights')
            biases = tf.get_variable('biases')
        return [kernel, biases]

    @layer
    def tanh(self, input, name):
        return tf.tanh(input, name=name)

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def lrelu(self, input, name, leak = 0.2):
        return tf.maximum(input, leak*input, name=name)


    @layer
    def sigmoid(self, input, name):
        return tf.nn.sigmoid(input, name=name)

    @layer
    def bn(self, input, n_out, name, phase_train=True):
        """
            Batch normalization on convolutional maps.
            Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
            Args:
                x:           Tensor, 4D BHWD input maps
                n_out:       integer, depth of input maps
                phase_train: boolean tf.Varialbe, true indicates training phase
                scope:       string, variable scope
            Return:
                normed:      batch-normalized maps
            """
        with tf.variable_scope(self.name + name):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            #mean, var = tf.cond(phase_train,
            #                    mean_var_with_update,
            #                    lambda: (ema.average(batch_mean), ema.average(batch_var)))
            mean, var = mean_var_with_update()

            normed = tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)
        return normed

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True, reuse=False):
        with tf.variable_scope(self.name + name, reuse=reuse) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))


            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable)
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            print(weights)
            print(biases)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                              [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)
        else:
            return tf.nn.softmax(input, name=name)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

    @layer
    def reshape(self, input, shape, name):
        return tf.reshape(input, shape, name=name)

    @layer
    def transpose(self, input, shape, name):
        return tf.transpose(input, shape, name=name)

    @layer
    def rename(self, input, name):
        return input

    @layer
    def add(self, input, name):
        assert len(input) == 2
        return tf.add(input[0], input[1], name=name)

    @layer
    def mul(self, input, num_out, name):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)

            weights = self.make_var('weights', [dim, num_out], init_weights)
            return tf.matmul(input, weights, name=name)

    def set_cost(self, cost):
        self.cost = cost

    def set_optimizer(self, optimizer):
        self.opt = optimizer

    @rnn_layer
    def rnn(self, input, n_hidden, n_layer, name, keep_prob = 1., time_major = False, only_last = False):
        state = None
        if isinstance(input, list):
            state = input[1]
            input = input[0]

        with tf.variable_scope(self.name + name) as scope:
            cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            # 다중 레이어 구성을 다음과 같이 아주 간단하게 만들 수 있습니다.
            if n_layer > 1:
                cell = tf.contrib.rnn.MultiRNNCell([cell] * n_layer)

            # tf.nn.dynamic_rnn 함수를 이용해 순환 신경망을 만듭니다.
            outputs, states = tf.nn.dynamic_rnn(cell, input,initial_state=state, dtype=tf.float32, time_major=time_major)

            if only_last:
                return outputs[:,-1], states
            else:
                return outputs, states