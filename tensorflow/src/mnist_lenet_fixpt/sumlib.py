import re
import tensorflow as tf

def image_input_summary(x):
  """Show the images in tensorboard"""
  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

def variable_summaries(var):
  # Attach a lot of summaries to a Tensor (for TensorBoard visualization).
  mean = tf.reduce_mean(var)
  tf.summary.scalar('mean', mean)
  with tf.name_scope('stddev'):
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
  tf.summary.scalar('stddev', stddev)
  tf.summary.scalar('max', tf.reduce_max(var))
  tf.summary.scalar('min', tf.reduce_min(var))
  tf.summary.histogram('histogram', var)


def fixed_point_conversion_summary(x, fixed_x, fix_def, acc):
  """Helper to create summaries for fixed point conversion steps.

  Creates a summary that provies a histogram of resulting tensor
  Creates a summary that provides the percentage innacuracy of the conversion

  Args:
    x: original tensor
    fixed_x: Resulting tensor
    fix_def: fix point definition for this conversion
    acc: accuracy array
  Returns:
    nothing
  """
  with tf.variable_scope('fix_def'):
    tf.summary.scalar('digit bits', fix_def[0])
    tf.summary.scalar('fraction bits', fix_def[1])

  with tf.variable_scope('acc'):
    tf.summary.scalar('percentage clip', (acc[0]))
    tf.summary.scalar('percentage under tolerance', (acc[1]))

  tf.summary.histogram('original', x)
  tf.summary.histogram('fixed', fixed_x)


def activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


