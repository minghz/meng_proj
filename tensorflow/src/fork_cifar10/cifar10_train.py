import tensorflow as tf
#from tensorflow.python import debug as tf_debug

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
  #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

  # Get images and labels for CIFAR-10.
  with tf.device('/cpu:0'):
    images = tf.placeholder(tf.float32,
            [FLAGS.batch_size,
            cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, 3],
            name='input-images')

    labels = tf.placeholder(tf.int32,
            [FLAGS.batch_size],
            name='input-labels')

  # Define all the fixed point variables we will be using later
  cifar10.initialize_fix_point_variables()

  # Build a Graph that computes the logits predictions from the inference model
  logits = cifar10.inference(images)

  # Calculate loss.
  loss = cifar10.loss(logits, labels)

  # Build a Graph that trains the model with one batch of examples and
  # updates the model parameters.
  train_op = cifar10.train(loss, 0.05)

  # Update fixed point conversion parameters when needed
  update_fix_pt_ops = cifar10.update_fix_point_accuracy()

  # Evaluate performance
  top_k_op = tf.nn.in_top_k(logits, labels, 1)

  # Merge all the summaries and write them out to
  # FLAGS.log_dir
  merged_summary = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

  # init all variables
  tf.global_variables_initializer().run()
  # needed on interactive session so it doesn't hang
  tf.train.start_queue_runners()

  def feed(train):
    print('in feed')
    if train: # training data
      print('in train')
      images, labels = cifar10.distorted_inputs()
    else: # testing data
      images, labels = cifar10.inputs(eval_data=eval_data)

    print('about to return')
    return {images: images.eval(session=sess), labels: labels.eval(session=sess)}

  for i in range(FLAGS.max_steps):
    print('before run')
    summary, _ = sess.run([merged_summary, train_op], feed_dict=feed(True))
    print('returned')
    train_writer.add_summary(summary, i) # summary
    print('wrote summary')

    if(i % 10 == 0):
      # updating fixed point accuracies
      sess.run([update_fix_pt_ops])
      print('Step: %s, Loss: %s' % (i, loss.eval()))

      # running tests to check accuracy
      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter:
        summary, predictions = sess.run([merged_summary, top_k_op], feed_dict=feed(False))
        true_count += np.sum(predictions)
        step += 1
  
      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
      
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      test_writer.add_summary(summary, i)

  train_writer.close()
  test_writer.close()


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  FLAGS = parser.parse_args()
  tf.app.run()
