import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class SP_LSTM_Cell(Layer):
    def __init__(self, units, input_dim, **kwargs):
        super(SP_LSTM_Cell, self).__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim
        self.state_size = [units, units]

    def build(self, input_shape):
        self.Wu = self.add_weight(shape=(self.input_dim + self.units, self.units), initializer='glorot_uniform')
        self.bu = self.add_weight(shape=(self.units,), initializer='zeros')

        self.Wf = self.add_weight(shape=(self.input_dim + self.units, self.units), initializer='glorot_uniform')
        self.bf = self.add_weight(shape=(self.units,), initializer='zeros')

        self.Wc = self.add_weight(shape=(self.input_dim + self.units, self.units), initializer='glorot_uniform')
        self.bc = self.add_weight(shape=(self.units,), initializer='zeros')

        self.Wo = self.add_weight(shape=(self.input_dim + self.units, self.units), initializer='glorot_uniform')
        self.bo = self.add_weight(shape=(self.units,), initializer='zeros')

        super(SP_LSTM_Cell, self).build(input_shape)

    def call(self, inputs, states):
        h_prev, c_prev = states
        x = tf.concat([inputs, h_prev], axis=1)

        u = tf.sigmoid(tf.matmul(x, self.Wu) + self.bu)
        f = tf.sigmoid(tf.matmul(x, self.Wf) + self.bf)
        c_tilde = tf.tanh(tf.matmul(x, self.Wc) + self.bc)
        c = u * c_tilde + f * c_prev
        o = tf.sigmoid(tf.matmul(x, self.Wo) + self.bo)
        h = o * tf.tanh(c)

        return h, [h, c]

class SP_LSTM(Layer):
    def __init__(self, units, input_dim, return_sequences=False, **kwargs):
        super(SP_LSTM, self).__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim
        self.return_sequences = return_sequences
        self.cell = SP_LSTM_Cell(units, input_dim)

    def call(self, inputs):
        h_prev = tf.zeros((tf.shape(inputs)[0], self.units))
        c_prev = tf.zeros((tf.shape(inputs)[0], self.units))
        outputs = []

        for t in range(inputs.shape[1]):
            output, [h_prev, c_prev] = self.cell(inputs[:, t, :], [h_prev, c_prev])
            outputs.append(output)

        if self.return_sequences:
            return tf.stack(outputs, axis=1)
        else:
            return outputs[-1]

    def get_config(self):
        config = super(SP_LSTM, self).get_config()
        config.update({
            'units': self.units,
            'input_dim': self.input_dim,
            'return_sequences': self.return_sequences
        })
        return config

