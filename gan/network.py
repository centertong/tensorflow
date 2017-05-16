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
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING,
             group=1, trainable=True, reuse = False):
        self.validate_padding(padding)
        c_i = input.get_shape().as_list()[-1]
        assert c_i % group == 0
        assert c_o % group == 0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(self.name + name, reuse=reuse) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i / group, c_o], init_weights,
                                   trainable)
            biases = self.make_var('biases', [c_o], init_biases, trainable)

            if group == 1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)
            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return tf.nn.relu(bias, name=scope.name)
            return tf.nn.bias_add(conv, biases, name=scope.name)

    @layer
    def deconv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING,
             trainable=True, reuse = False):
        c_n = -1
        c_h = input.get_shape().as_list()[1] * s_h
        c_w = input.get_shape().as_list()[2] * s_w
        c_i = input.get_shape().as_list()[-1]
        deconvolve = lambda i, k: tf.nn.conv2d_transpose(i, k, output_shape= [c_n, c_h, c_w, c_o],strides= [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(self.name + name, reuse= reuse) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.02)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_o, c_i], init_weights,
                                   trainable)
            biases = self.make_var('biases', [c_o], init_biases, trainable)

            deconv = deconvolve(input, kernel)

            if relu:
                bias = tf.nn.bias_add(deconv, biases)
                return tf.nn.relu(bias, name=scope.name)
            return tf.nn.bias_add(deconv, biases, name=scope.name)

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
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)


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

    def set_cost_gen(self, cost):
        self.cost_gen = cost

    def set_cost_dis(self, cost):
        self.cost_dis = cost

    def set_optimizer_gen(self, optimizer):
        self.opt_gen = optimizer

    def set_optimizer_dis(self, optimizer):
        self.opt_dis = optimizer

    def inception(self, input, c_conv1, c_conv2_1, c_conv2_2, c_conv3_1, c_conv3_2, c_conv4, name):
        # inceiption3a
        (self.feed(input)
         .conv(1, 1, c_conv1, 1, 1, name=name + '_conv1'))
        (self.feed(input)
         .conv(1, 1, c_conv2_1, 1, 1, name=name + '_conv2_1')
         .conv(3, 3, c_conv2_2, 1, 1, name=name + '_conv2_2'))
        (self.feed(input)
         .conv(1, 1, c_conv3_1, 1, 1, name=name + '_conv3_1')
         .conv(5, 5, c_conv3_2, 1, 1, name=name + '_conv3_2'))
        (self.feed(input)
         .max_pool(3, 3, 1, 1, name=name + '_pool')
         .conv(1, 1, c_conv4, 1, 1, name=name + '_conv3'))
        (self.feed(name + '_conv1', name + '_conv2_2', name + '_conv3_2', name + '_conv3')
         .concat(3, name=name))

    def residual_v1(self, input, c_conv1, c_conv2, k, name):
        (self.feed(input)
         .conv(3,3, c_conv1, k,k, name= name + '_conv1')
         .conv(3,3, c_conv2, 1,1, name= name + '_conv2'))
        if k == 1:
            (self.feed(input, name + '_conv2').add(name=name))
        else:
            (self.feed(input).mul(c_conv2, name= name+'_intermediate'))
            (self.feed(name+ '_intermediate', name + '_conv2').add(name=name))


    def residual_v2(self, input, c_conv1, c_conv2, c_conv3, k, name):
        (self.feed(input)
         .conv(1, 1, c_conv1, k, k, name=name + '_conv1')
         .conv(3, 3, c_conv2, 1, 1, name=name + '_conv2')
         .conv(1, 1, c_conv3, 1, 1, name=name + '_conv3'))

        if k == 1:
            (self.feed(input, name + '_conv2').add(name=name))
        else:
            (self.feed(input).mul(c_conv3, name=name + '_intermediate'))
            (self.feed(name + '_intermediate', name + '_conv2').add(name=name))

