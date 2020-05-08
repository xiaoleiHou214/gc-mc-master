from __future__ import print_function


from gcmc.initializations import *
import tensorflow as tf

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)

    return pre_out * tf.div(1., keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    # Properties
        name: String, defines the variable scope of the layer.
            Layers with common name share variables. (TODO)
        logging: Boolean, switches Tensorflow histogram logging on/off
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer for two types of nodes in a bipartite graph. """

    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, share_user_item_weights=False,
                 bias=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        with tf.variable_scope(self.name + '_vars'):
            if not share_user_item_weights:

                self.vars['weights_u'] = weight_variable_random_uniform(input_dim, output_dim, name="weights_u")
                self.vars['weights_v'] = weight_variable_random_uniform(input_dim, output_dim, name="weights_v")

                if bias:
                    self.vars['user_bias'] = bias_variable_truncated_normal([output_dim], name="bias_u")
                    self.vars['item_bias'] = bias_variable_truncated_normal([output_dim], name="bias_v")


            else:
                self.vars['weights_u'] = weight_variable_random_uniform(input_dim, output_dim, name="weights")
                self.vars['weights_v'] = self.vars['weights_u']

                if bias:
                    self.vars['user_bias'] = bias_variable_truncated_normal([output_dim], name="bias_u")
                    self.vars['item_bias'] = self.vars['user_bias']

        self.bias = bias

        self.dropout = dropout
        self.act = act
        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x_u = inputs[0]
        x_u = tf.nn.dropout(x_u, 1 - self.dropout)
        x_u = tf.matmul(x_u, self.vars['weights_u'])

        x_v = inputs[1]
        x_v = tf.nn.dropout(x_v, 1 - self.dropout)
        x_v = tf.matmul(x_v, self.vars['weights_v'])

        u_outputs = self.act(x_u)
        v_outputs = self.act(x_v)

        if self.bias:
            u_outputs += self.vars['user_bias']
            v_outputs += self.vars['item_bias']

        return u_outputs, v_outputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging:
                tf.summary.histogram(self.name + '/inputs_u', inputs[0])
                tf.summary.histogram(self.name + '/inputs_v', inputs[1])
            outputs_u, outputs_v = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs_u', outputs_u)
                tf.summary.histogram(self.name + '/outputs_v', outputs_v)
            return outputs_u, outputs_v


class StackGCN(Layer):
    """Graph convolution layer for bipartite graphs and sparse inputs."""

    def __init__(self, input_dim, output_dim, num_users, num_items, support, support_t, num_support, u_features_nonzero=None,
                 v_features_nonzero=None, sparse_inputs=False, dropout=0.,
                 act=tf.nn.relu, share_user_item_weights=True, **kwargs):
        super(StackGCN, self).__init__(**kwargs)

        assert output_dim % num_support == 0, 'output_dim must be multiple of num_support for stackGC layer'

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_u'] = weight_variable_random_uniform(input_dim, output_dim, name='weights_u')

            if not share_user_item_weights:
                self.vars['weights_v'] = weight_variable_random_uniform(input_dim, output_dim, name='weights_v')

            else:
                self.vars['weights_v'] = self.vars['weights_u']


        self.weights_u = tf.split(value=self.vars['weights_u'], axis=1, num_or_size_splits=num_support)
        self.weights_v = tf.split(value=self.vars['weights_v'], axis=1, num_or_size_splits=num_support)

        # TODO: add attention Layer weight
        hidden_size = int(output_dim / num_support)
        attention_size = 64
        omega_size_u = num_users
        omega_size_v = num_items
        for i in range(num_support):
            # self.vars['w_omega_%s' % ('u' + str(i))] = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))  # E * T
            # self.vars['u_omega_%s' % ('u' + str(i))] = tf.Variable(tf.random_normal([omega_size_u], stddev=0.1))  # T
            # self.vars['w_omega_%s' % ('v' + str(i))] = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))  # E * T
            # self.vars['u_omega_%s' % ('v' + str(i))] = tf.Variable(tf.random_normal([omega_size_v], stddev=0.1))  # T
            self.vars['w_omega_%s' % ('u' + str(i))] = weight_variable_random_uniform(hidden_size, attention_size, name='w_omega_%s' % ('u' + str(i)))  # E * T
            self.vars['u_omega_%s' % ('u' + str(i))] = weight_variable_random_uniform(num_items,num_users, name='u_omega_%s' % ('u' + str(i)))  # T
            self.vars['w_omega_%s' % ('v' + str(i))] = weight_variable_random_uniform(hidden_size, attention_size, name='w_omega_%s' % ('v' + str(i)))  # E * T
            self.vars['u_omega_%s' % ('v' + str(i))] = weight_variable_random_uniform(num_users,num_items, name='u_omega_%s' % ('v' + str(i)))
        self.dropout = dropout
        # TODO: add
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_users = num_users
        self.num_items = num_items

        self.sparse_inputs = sparse_inputs
        self.u_features_nonzero = u_features_nonzero
        self.v_features_nonzero = v_features_nonzero
        if sparse_inputs:
            assert u_features_nonzero is not None and v_features_nonzero is not None, \
                'u_features_nonzero and v_features_nonzero can not be None when sparse_inputs is True'

        self.support = tf.sparse_split(axis=1, num_split=num_support, sp_input=support)
        self.support_transpose = tf.sparse_split(axis=1, num_split=num_support, sp_input=support_t)
        self.act = act

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x_u = inputs[0]
        x_v = inputs[1]

        if self.sparse_inputs:
            x_u = dropout_sparse(x_u, 1 - self.dropout, self.u_features_nonzero)
            x_v = dropout_sparse(x_v, 1 - self.dropout, self.v_features_nonzero)
        else:
            x_u = tf.nn.dropout(x_u, 1 - self.dropout)
            x_v = tf.nn.dropout(x_v, 1 - self.dropout)

        supports_u = []
        supports_v = []

        for i in range(len(self.support)):
            # TODO: q(i,j)
            tmp_u = dot(x_u, self. weights_u[i], sparse=self.sparse_inputs)
            tmp_v = dot(x_v, self.weights_v[i], sparse=self.sparse_inputs)

            support = self.support[i]
            support_transpose = self.support_transpose[i]

            print('$' * 50)
            print('#' * 50)
            print(self.num_users)
            print(self.num_items)
            support_dense = tf.sparse_to_dense(support.indices, [self.num_users, self.num_items], support.values,
                                               validate_indices=False, default_value=0.0)
            support_dense_binary = self.getBinaryTensor(support_dense)
            support_t_dense = tf.sparse_to_dense(support_transpose.indices, [self.num_items, self.num_users],
                                                 support_transpose.values, default_value=0.0, validate_indices=False)
            support_t_dense_binary = self.getBinaryTensor(support_t_dense)

            # TODO: add Attention Layer
            atten_alphas_u = self.attention(tmp_u, support_t_dense_binary, 'u' + str(i), self.num_users, self.num_items)    # M*N
            atten_alphas_v = self.attention(tmp_v, support_dense_binary, 'v' + str(i), self.num_items, self.num_users)      # N*M

            alphas_support_t = tf.multiply(support_t_dense, atten_alphas_u)       # M*N
            alphas_support = tf.multiply(support_dense, atten_alphas_v)           # N*M
            # supports_u.append(tf.matmul(atten_alphas_v, tmp_v))
            # supports_v.append(tf.matmul(atten_alphas_u, tmp_u))
            supports_u.append(tf.sparse_tensor_dense_matmul(support, tmp_v))
            supports_v.append(tf.sparse_tensor_dense_matmul(support_transpose, tmp_u))

        z_u = tf.concat(axis=1, values=supports_u)
        z_v = tf.concat(axis=1, values=supports_v)

        u_outputs = self.act(z_u)
        v_outputs = self.act(z_v)
        print('#'*50)
        print('#' * 50)

        return u_outputs, v_outputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs_u', inputs[0])
                tf.summary.histogram(self.name + '/inputs_v', inputs[1])
            outputs_u, outputs_v = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs_u', outputs_u)
                tf.summary.histogram(self.name + '/outputs_v', outputs_v)
            return outputs_u, outputs_v

    def attention(self, inputs, support, la, num1, num2):

        # input: N * E
        attention_size = 64

        s_m = tf.matmul(support, inputs)    # M*E   support:[M*N]
        sm_num = tf.reduce_sum(support, axis=1)     # M*1
        sm_mean = tf.divide(1, sm_num)
        s_m = tf.multiply(sm_mean, tf.transpose(s_m))       # M * e*M
        s_m = tf.transpose(s_m)     # M*E
        # TODO: W * sm
        inputs_t = tf.transpose(inputs)     # M*N
        w_sm = tf.tensordot(s_m, self.vars['w_omega_%s' % (la)], axes=1)    # [M*E] * [E*A] = [M*A]
        # TODO: modifiy
        support_3d = tf.reshape(support, [num2, 1, num1])       # M*N
        s_j_all = tf.transpose(tf.multiply(support_3d, inputs_t), [0,2,1])      # M*1*N * [E*N] = [M*E*N]
        #w_sj_all = tf.tensordot(s_j_all, self.vars['w_omega_%s' % (la)], axes=1)        # [M*N*E] * [E*A]
        w_sj_all = tf.tensordot(s_j_all, self.vars['w_omega_%s' % (la)], axes=1)
        # M*A - M*N*A = M*N*A M*N
        w_sm = tf.reshape(w_sm, [num2, 1, attention_size])      # M*1*A
        euclidean_set = tf.sqrt(tf.reduce_sum(tf.square(w_sm - w_sj_all), 2))       # [M*1*A] * [M*N*A]
        eucl = tf.tanh(euclidean_set)     # M*N
        print("euclidean computing success!")
        print(eucl.shape)

        vu = tf.multiply(self.vars['u_omega_%s' % (la)], eucl)
        # vu = tf.matmul(self.vars['u_omega_%s' % (la)], eucl)
        #vu = tf.tensordot(eucl,self.vars['u_omega_%s' % (la)],axes=1)
        print("vu computing success!")

        alphas = tf.nn.softmax(vu, name='alphas_%s' % (la))  # (B,T) shape
        print("alphas computing success!")

        return alphas

    def attention2(self, inputs, support, la, num1, num2):

        # input: N * E
        attention_size = 64

        support = tf.ones_like(support)
        s_m = tf.matmul(support, inputs)    # M*E   support:[M*N]
        sm_num = tf.reduce_sum(support, axis=1)     # M*1
        sm_mean = tf.divide(1, sm_num)
        s_m = tf.multiply(sm_mean, tf.transpose(s_m))       # M * e*M
        s_m = tf.transpose(s_m)     # M*E
        # TODO: W * sm
        inputs_t = tf.transpose(inputs)     # M*N
        w_sm = tf.tensordot(s_m, self.vars['w_omega_%s' % (la)], axes=1)    # [M*E] * [E*A] = [M*A]
        # TODO: modifiy
        support_3d = tf.reshape(support, [num2, 1, num1])       # M*N
        s_j_all = tf.transpose(tf.multiply(support_3d, inputs_t), [0,2,1])      # M*1*N * [E*N] = [M*E*N]
        w_sj_all = tf.tensordot(s_j_all, self.vars['w_omega_%s' % (la)], axes=1)        # [M*N*E] * [E*A]
        # M*A - M*N*A = M*N*A M*N
        w_sm = tf.reshape(w_sm, [num2, 1, attention_size])      # M*1*A
        #euclidean_set = tf.sqrt(tf.reduce_sum(tf.square(w_sm - w_sj_all), 2))       # [M*1*A] * [M*N*A]
        euclidean_set = w_sm - w_sj_all
        eucl = tf.tanh(euclidean_set)     # M*N
        print("euclidean computing success!")
        print(eucl.shape)

        # vu = tf.multiply(self.vars['u_omega_%s' % (la)], eucl)
        # vu = tf.matmul(self.vars['u_omega_%s' % (la)], eucl)
        vu = tf.tensordot(eucl,self.vars['u_omega_%s' % (la)],axes=1)
        print("vu computing success!")

        alphas = tf.nn.softmax(vu, name='alphas_%s' % (la))  # (B,T) shape
        print("alphas computing success!")

        return alphas

    def attention3(self, inputs, support, la, num1, num2):

        # input: N * E
        attention_size = 64

        s_m = tf.sparse_tensor_dense_matmul(support, inputs)    # M*E   support:[M*N]
        sm_num = tf.sparse_reduce_sum(support, axis=1)     # M*1
        sm_mean = tf.divide(1, sm_num)
        s_m = tf.multiply(sm_mean, tf.transpose(s_m))       # M * e*M
        s_m = tf.transpose(s_m)     # M*E
        # TODO: W * sm
        inputs_t = tf.transpose(inputs)     # M*N
        w_sm = tf.tensordot(s_m, self.vars['w_omega_%s' % (la)], axes=1)    # [M*E] * [E*A] = [M*A]
        # TODO: modifiy
        support_3d = tf.sparse_reshape(support, [num2, 1, num1])       # M*N

        support_3d = tf.sparse_to_dense(support_3d.indices, [num2, 1, num1], support_3d.values,
                                        validate_indices=False, default_value=0.0)

        s_j_all = tf.transpose(tf.multiply(support_3d, inputs_t), [0,2,1])      # M*1*N * [E*N] = [M*E*N]
        w_sj_all = tf.tensordot(s_j_all, self.vars['w_omega_%s' % (la)], axes=1)        # [M*N*E] * [E*A]
        # M*A - M*N*A = M*N*A M*N
        w_sm = tf.reshape(w_sm, [num2, 1, attention_size])      # M*1*A
        euclidean_set = tf.sqrt(tf.reduce_sum(tf.square(w_sm - w_sj_all), 2))       # [M*1*A] * [M*N*A]
        eucl = tf.tanh(euclidean_set)     # M*N
        print("euclidean computing success!")
        print(eucl.shape)

        vu = tf.multiply(self.vars['u_omega_%s' % (la)], eucl)
        # vu = tf.matmul(self.vars['u_omega_%s' % (la)], eucl)
        #vu = tf.tensordot(eucl,self.vars['u_omega_%s' % (la)],axes=1)
        print("vu computing success!")

        alphas = tf.nn.softmax(vu, name='alphas_%s' % (la))  # (B,T) shape
        print("alphas computing success!")

        return alphas

    def getBinaryTensor(self, supportTensor, boundary=0.0):
        one = tf.ones_like(supportTensor)
        zero = tf.zeros_like(supportTensor)
        return tf.where(supportTensor != boundary, one, zero)


class OrdinalMixtureGCN(Layer):

    """Graph convolution layer for bipartite graphs and sparse inputs."""

    def __init__(self, input_dim, output_dim, support, support_t, num_support, u_features_nonzero=None,
                 v_features_nonzero=None, sparse_inputs=False, dropout=0.,
                 act=tf.nn.relu, bias=False, share_user_item_weights=False, self_connections=False, **kwargs):
        super(OrdinalMixtureGCN, self).__init__(**kwargs)

        with tf.variable_scope(self.name + '_vars'):

            self.vars['weights_u'] = tf.stack([weight_variable_random_uniform(input_dim, output_dim,
                                                                             name='weights_u_%d' % i)
                                              for i in range(num_support)], axis=0)

            if bias:
                self.vars['bias_u'] = bias_variable_const([output_dim], 0.01, name="bias_u")

            if not share_user_item_weights:
                self.vars['weights_v'] = tf.stack([weight_variable_random_uniform(input_dim, output_dim,
                                                                                 name='weights_v_%d' % i)
                                                  for i in range(num_support)], axis=0)

                if bias:
                    self.vars['bias_v'] = bias_variable_const([output_dim], 0.01, name="bias_v")

            else:
                self.vars['weights_v'] = self.vars['weights_u']
                if bias:
                    self.vars['bias_v'] = self.vars['bias_u']

        self.weights_u = self.vars['weights_u']
        self.weights_v = self.vars['weights_v']

        self.dropout = dropout

        self.sparse_inputs = sparse_inputs
        self.u_features_nonzero = u_features_nonzero
        self.v_features_nonzero = v_features_nonzero
        if sparse_inputs:
            assert u_features_nonzero is not None and v_features_nonzero is not None, \
                'u_features_nonzero and v_features_nonzero can not be None when sparse_inputs is True'

        self.self_connections = self_connections

        self.bias = bias
        support = tf.sparse_split(axis=1, num_split=num_support, sp_input=support)

        support_t = tf.sparse_split(axis=1, num_split=num_support, sp_input=support_t)

        if self_connections:
            self.support = support[:-1]
            self.support_transpose = support_t[:-1]
            self.u_self_connections = support[-1]
            self.v_self_connections = support_t[-1]
            self.weights_u = self.weights_u[:-1]
            self.weights_v = self.weights_v[:-1]
            self.weights_u_self_conn = self.weights_u[-1]
            self.weights_v_self_conn = self.weights_v[-1]

        else:
            self.support = support
            self.support_transpose = support_t
            self.u_self_connections = None
            self.v_self_connections = None
            self.weights_u_self_conn = None
            self.weights_v_self_conn = None

        self.support_nnz = []
        self.support_transpose_nnz = []
        for i in range(len(self.support)):
            nnz = tf.reduce_sum(tf.shape(self.support[i].values))
            self.support_nnz.append(nnz)
            self.support_transpose_nnz.append(nnz)

        self.act = act

        if self.logging:
            self._log_vars()

    def _call(self, inputs):

        if self.sparse_inputs:
            x_u = dropout_sparse(inputs[0], 1 - self.dropout, self.u_features_nonzero)
            x_v = dropout_sparse(inputs[1], 1 - self.dropout, self.v_features_nonzero)
        else:
            x_u = tf.nn.dropout(inputs[0], 1 - self.dropout)
            x_v = tf.nn.dropout(inputs[1], 1 - self.dropout)

        supports_u = []
        supports_v = []

        # self-connections with identity matrix as support
        if self.self_connections:
            uw = dot(x_u, self.weights_u_self_conn, sparse=self.sparse_inputs)
            supports_u.append(tf.sparse_tensor_dense_matmul(self.u_self_connections, uw))

            vw = dot(x_v, self.weights_v_self_conn, sparse=self.sparse_inputs)
            supports_v.append(tf.sparse_tensor_dense_matmul(self.v_self_connections, vw))

        wu = 0.
        wv = 0.
        for i in range(len(self.support)):
            wu += self.weights_u[i]
            wv += self.weights_v[i]

            # multiply feature matrices with weights
            tmp_u = dot(x_u, wu, sparse=self.sparse_inputs)

            tmp_v = dot(x_v, wv, sparse=self.sparse_inputs)

            support = self.support[i]
            support_transpose = self.support_transpose[i]

            # then multiply with rating matrices
            supports_u.append(tf.sparse_tensor_dense_matmul(support, tmp_v))
            supports_v.append(tf.sparse_tensor_dense_matmul(support_transpose, tmp_u))

        z_u = tf.add_n(supports_u)
        z_v = tf.add_n(supports_v)

        if self.bias:
            z_u = tf.nn.bias_add(z_u, self.vars['bias_u'])
            z_v = tf.nn.bias_add(z_v, self.vars['bias_v'])

        u_outputs = self.act(z_u)
        v_outputs = self.act(z_v)

        return u_outputs, v_outputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs_u', inputs[0])
                tf.summary.histogram(self.name + '/inputs_v', inputs[1])
            outputs_u, outputs_v = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs_u', outputs_u)
                tf.summary.histogram(self.name + '/outputs_v', outputs_v)
            return outputs_u, outputs_v


class BilinearMixture(Layer):
    """
    Decoder model layer for link-prediction with ratings
    To use in combination with bipartite layers.
    """

    def __init__(self, num_classes, u_indices, v_indices, input_dim, num_users, num_items, user_item_bias=False,
                 dropout=0., act=tf.nn.softmax, num_weights=3,
                 diagonal=True, **kwargs):
        super(BilinearMixture, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):

            for i in range(num_weights):
                if diagonal:
                    #  Diagonal weight matrices for each class stored as vectors
                    self.vars['weights_%d' % i] = weight_variable_random_uniform(1, input_dim, name='weights_%d' % i)

                else:
                    self.vars['weights_%d' % i] = orthogonal([input_dim, input_dim], name='weights_%d' % i)

            self.vars['weights_scalars'] = weight_variable_random_uniform(num_weights, num_classes,
                                                                          name='weights_u_scalars')

            if user_item_bias:
                self.vars['user_bias'] = bias_variable_zero([num_users, num_classes], name='user_bias')
                self.vars['item_bias'] = bias_variable_zero([num_items, num_classes], name='item_bias')

        self.user_item_bias = user_item_bias

        if diagonal:
            self._multiply_inputs_weights = tf.multiply
        else:
            self._multiply_inputs_weights = tf.matmul

        self.num_classes = num_classes
        self.num_weights = num_weights
        self.u_indices = u_indices
        self.v_indices = v_indices

        self.dropout = dropout
        self.act = act
        if self.logging:
            self._log_vars()

    def _call(self, inputs):

        u_inputs = tf.nn.dropout(inputs[0], 1 - self.dropout)
        v_inputs = tf.nn.dropout(inputs[1], 1 - self.dropout)

        u_inputs = tf.gather(u_inputs, self.u_indices)
        v_inputs = tf.gather(v_inputs, self.v_indices)

        if self.user_item_bias:
            u_bias = tf.gather(self.vars['user_bias'], self.u_indices)
            v_bias = tf.gather(self.vars['item_bias'], self.v_indices)
        else:
            u_bias = None
            v_bias = None

        basis_outputs = []
        for i in range(self.num_weights):

            u_w = self._multiply_inputs_weights(u_inputs, self.vars['weights_%d' % i])
            x = tf.reduce_sum(tf.multiply(u_w, v_inputs), axis=1)

            basis_outputs.append(x)

        # Store outputs in (Nu x Nv) x num_classes tensor and apply activation function
        basis_outputs = tf.stack(basis_outputs, axis=1)

        outputs = tf.matmul(basis_outputs,  self.vars['weights_scalars'], transpose_b=False)

        if self.user_item_bias:
            outputs += u_bias
            outputs += v_bias

        outputs = self.act(outputs)

        return outputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs_u', inputs[0])
                tf.summary.histogram(self.name + '/inputs_v', inputs[1])

            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

def arr2sparse(tensor):
    arr_idx = tf.where(tf.not_equal(tensor, 0))
    arr_sparse = tf.SparseTensor(arr_idx, tf.gather_nd(tensor, arr_idx), tensor.get_shape())
    return arr_sparse