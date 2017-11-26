#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import mnist_lenet

from tensorflow.examples.tutorials.mnist import input_data

tf.NoGradient("ReshapeFix")
# use custom op to calculate gradients
reshape_fix = tf.load_op_library('./custom_ops/reshape_fix.so').reshape_fix

# supress tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = None

def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)
  sess = tf.InteractiveSession()

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
  mnist_lenet.image_input_summary(x)

  # reshape input to 4d tensor
  # -1?, 28x28 widthxheight, 1 color channel
  flat_inputs = tf.reshape(x, [-1, 28, 28, 1])

  # First conv. layer has 32 features of 5x5 patch
  # 5x5 patch, 1 input chanel, 32 ouput chanel (features)
  # first conv layer, output is same as input 28x28
  # first max pool layer, endinv up with 14x14 size
  conv1 = mnist_lenet.conv_layer([5, 5], 1, 32, flat_inputs, 'conv1')

  # second conv layer, 64 features, 5x5 patch
  # 32 input channel because 32 input feature from 1st layer
  conv2 = mnist_lenet.conv_layer([5, 5], 32, 64, conv1, 'conv2')

  # -1?, flatten the output from 2nd layer
  flat_conv2 = tf.reshape(conv2, [-1, 7 * 7 * 64])

  # add fully-connected layer of 1024 neurons to process everything
  # one dimention the output from 2nd layer, 1024 neurons
  # ( x*W + b ) line normal fully connected
  local3 = mnist_lenet.nn_layer(flat_conv2, 7 * 7 * 64, 1024, 'local3')

  # Apply dropout technique to avoid overfitting
  local3_drop, keep_prob = mnist_lenet.dropout(local3)

  # output layer, one-hot
  local4 = mnist_lenet.nn_layer(local3_drop, 1024, 10, 'local4')

  # cross_entropy == loss
  loss = mnist_lenet.cross_entropy(y_, local4)
  tf.summary.scalar('loss', loss)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(local4, 1), tf.argmax(y_, 1))
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
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()

  print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images,
                                                      y_: mnist.test.labels,
                                                      keep_prob: 1.0}))

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
