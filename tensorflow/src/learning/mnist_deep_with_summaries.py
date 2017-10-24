# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.

This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.

It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# supress tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = None


def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)
  sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

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

  # convolution layer - output same size as input: stride = 1; 0-padded
  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  # pooling layer - max pooling over 2x2 blocks
  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # First conv. layer has 32 features of 5x5 patch
  # 5x5 patch, 1 input chanel, 32 ouput chanel (features)
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])  # one bias for each of 32 channels (features)

  # reshape input to 4d tensor
  # -1?, 28x28 widthxheight, 1 color channel
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # first conv layer, output is same as input 28x28
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  # first max pool layer, endinv up with 14x14 size
  h_pool1 = max_pool_2x2(h_conv1)

  # second conv layer, 64 features, 5x5 patch
  # 32 input channel because 32 input feature from 1st layer
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)  # should have size 7x7 here

  # add fully-connected layer of 1024 neurons to process everything
  # one dimention the output from 2nd layer, 1024 neurons
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  # -1?, flatten the output from 2nd layer
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
  # ( x*W + b ) line normal fully connected
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Apply Dropout to avoid overfitting
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    # raw outputs of the nn_layer above, and then average across
    # the batch.
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y_conv, 1),
                                      tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  # FLAGS.log_dir
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(50, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 100 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 1000 == 999:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True), options=run_options, run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()

  print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=20000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.5,
                      help='Keep probability for training dropout.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('VIRTUAL_ENV', '.'),
                           'input_data/mnist'),
      help='Directory for storing input data')
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('SUMMARY_DIR', '.'),
                           'summaries', sys.argv[0].split('.py')[0]),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
