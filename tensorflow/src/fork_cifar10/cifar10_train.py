import tensorflow as tf

import os
import cifar10 # library of functions

# supress tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# use custom op to calculate gradients
reshape_fix = tf.load_op_library('custom_ops/reshape_fix.so').reshape_fix

parser = cifar10.parser

def train():
  """Train CIFAR-10 for a number of steps."""
  sess = tf.InteractiveSession()

  # Get images and labels for CIFAR-10.

  with tf.device('/cpu:0'):
    images, labels = cifar10.distorted_inputs()

  # Build a Graph that computes the logits predictions from the inference model
  logits = cifar10.inference(images)

  # Calculate loss.
  loss = cifar10.loss(logits, labels)

  # Build a Graph that trains the model with one batch of examples and
  # updates the model parameters.
  train_op = cifar10.train(loss, 0.05)

  # Merge all the summaries and write them out to
  # FLAGS.log_dir
  merged_summary = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

  # init all variables
  tf.global_variables_initializer().run()
  # needed on interactive session so it doesn't hang
  tf.train.start_queue_runners()

  for i in range(FLAGS.max_steps):
    summary, _ = sess.run([merged_summary, train_op])
    train_writer.add_summary(summary, i) # summary

    if(i % 10 == 0):
      print('Step: %s, Loss: %s' % (i, loss.eval()))

  train_writer.close()

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  FLAGS = parser.parse_args()
  tf.app.run()
