#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

def initialize_variables():
  """Initializing variables:
  1. Conversion Tensors that sit in-between layers
  2. Fixed Point number definitions for leyers in 1)
  3. Weight and Bias tensors for each layer
  4. Fixed Point number definitions for layers in 3)
  """
  # The output of the major layers that we want to convert to fixed point precision
  scope_names = ['input', 'conv1', 'conv2', 'local3', 'local4']
  for scope_name in scope_names:
    with tf.variable_scope(scope_name):
      # Fixed point nunmber precision settings
      fix_def = tf.get_variable('fix_def', initializer=[1, 1], trainable=False)

      # Fixed point precision adjustments
      # Assigning more or less digit/fraction bits to fixed point number
      acc = tf.get_variable('acc', initializer=[0., 0.], trainable=False)

# Initialize weights with a small value to prevent 0 gradients
def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def variable_summaries(var):
  # Attach a lot of summaries to a Tensor (for TensorBoard visualization).
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

# convolution layer - output same size as input: stride = 1; 0-padded
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# pooling layer - max pooling over 2x2 blocks
def max_pool_2x2(x):
  return tf.nn.max_pool(x,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')

def conv_layer(patch_dim,
               num_input_ch,
               num_features,
               flat_inputs,
               layer_name,
               act=tf.nn.relu):
  """ Reusable code for making a convolution layer
  It has a conv layer and a pooling layer
  """
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      W_conv = weight_variable([patch_dim[0],
                               patch_dim[1],
                               num_input_ch,
                               num_features])
      variable_summaries(W_conv)
    with tf.name_scope('biases'):
      b_conv = bias_variable([num_features])
      variable_summaries(b_conv)
    with tf.name_scope('conv'):
      h_conv = act(conv2d(flat_inputs, W_conv) + b_conv)
      tf.summary.histogram('convolutions', h_conv)
    h_pool = max_pool_2x2(h_conv)
    tf.summary.histogram('pools', h_pool)
    return h_pool