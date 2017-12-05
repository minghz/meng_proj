import tensorflow as tf
from tensorflow.python.framework import ops

# Defining custom operations
rf = tf.load_op_library('./custom_ops/reshape_fix.so')
reshape_fix = rf.reshape_fix
# Gradient registration for out custom operation
@ops.RegisterGradient("ReshapeFix")
def _reshape_fix_grad(op, grad):
  return rf.reshape_fix_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2])

import varlib
import sumlib

def to_fixed_point(x, scope):
  """Helper method to convert tensors to fixed point accuracy

  Args:
    x: input tensor
    scope: variable scope that is being converted
  Returns:
    fixed point accuracy equivalent tensor
  """
  with tf.variable_scope(scope):
    fix_def = tf.get_variable('fix_def', initializer=[1, 1], dtype=tf.int32, trainable=False)
    acc = tf.get_variable('acc', initializer=[0., 0.], trainable=False)

  fixed_x = reshape_fix(x, fix_def, acc)
  sumlib.fixed_point_conversion_summary(x, fixed_x, fix_def, acc)

  return fixed_x

def update_fix_point_accuracy():
  """Update the fixed point variable accuracy if required

  Returns:
    update_ops: list of operations that updates the fix_def of all layers
  """
  update_ops = []
  scope_names = ['conv1', 'conv2', 'local3', 'local4']
  for scope_name in scope_names:
    with tf.variable_scope(scope_name, reuse=True):
      fix_def = tf.get_variable('fix_def', [2], dtype=tf.int32)
      acc = tf.get_variable('acc', [2])

      fix_def = tf.cond(
        acc[0] > 0.05, # overflow clipped
        lambda: tf.assign_add(fix_def, [1, 0]),
        lambda: tf.assign_add(fix_def, [0, 0])
      )

      fix_def = tf.cond(
        acc[1] > 0.05, # under tolerance
          lambda: tf.assign_add(fix_def, [0, 1]),
          lambda: tf.assign_add(fix_def, [0, 0])
      )
      update_ops.append(fix_def)

  return update_ops


def initialize_variables():
  """Initializing variables:
  1. Conversion Tensors that sit in-between layers
  2. Fixed Point number definitions for leyers in 1)
  3. Weight and Bias tensors for each layer
  4. Fixed Point number definitions for layers in 3)
  """
  # The output of the major layers that we want to convert to fixed point precision
  scope_names = ['conv1', 'conv2', 'local3', 'local4']
  for scope_name in scope_names:
    with tf.variable_scope(scope_name):
      # Fixed point nunmber precision settings
      fix_def = tf.get_variable('fix_def', initializer=[1, 1], trainable=False)

      # Fixed point precision adjustments
      # Assigning more or less digit/fraction bits to fixed point number
      acc = tf.get_variable('acc', initializer=[0., 0.], trainable=False)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  # This Variable will hold the state of the weights for the layer
  # weights
  with tf.name_scope('weights'):
    weights = varlib.weight_variable([input_dim, output_dim])
    sumlib.variable_summaries(weights)
  # biases
  with tf.name_scope('biases'):
    biases = varlib.bias_variable([output_dim])
    sumlib.variable_summaries(biases)
  # Wx_+_b
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
  # wieghts
  with tf.name_scope('weights'):
    W_conv = varlib.weight_variable([patch_dim[0],
                             patch_dim[1],
                             num_input_ch,
                             num_features])
    sumlib.variable_summaries(W_conv)
  # biases
  with tf.name_scope('biases'):
    b_conv = varlib.bias_variable([num_features])
    sumlib.variable_summaries(b_conv)
  # activation convolution
  with tf.name_scope('activation'):
    h_conv = act(conv2d(flat_inputs, W_conv) + b_conv)
    tf.summary.histogram('convolutions', h_conv)
  # max pool
  with tf.name_scope('max_pool'):
    h_pool = max_pool_2x2(h_conv)
    tf.summary.histogram('pools', h_pool)
  return h_pool

def dropout(local3):
  """Apply Dropout to avoid overfitting
  note: its happening between the last fully connected layer
  and the output layer"""

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    return tf.nn.dropout(local3, keep_prob), keep_prob

def cross_entropy(y_, local4):
  """Cross entropy is also the technique we are using to figure out
  the loss of the model"""

  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=local4)
    with tf.name_scope('total'):
      return tf.reduce_mean(diff)
