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

