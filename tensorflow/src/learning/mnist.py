#
#
# Fully Connected Network model to solve MNIST problem
# Following example of TensorFlow tutorials
#
#

# load mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

# interactive session
import tensorflow as tf
sess = tf.InteractiveSession()

# INITIALIZE ===================================================================
# input image in pixels
x = tf.placeholder(tf.float32, shape=[None, 784])
# output in class [0, 9]
y_ = tf.placeholder(tf.float32, shape=[None, 10])
#OBS: 'None' means batch size is any size

# variables initialized to zero
# W is a 784x10 matrix because we have 784 input features and 10 outputs
W = tf.Variable(tf.zeros([784, 10]))
# b is 10 dimentional vector (10 classes)
b = tf.Variable(tf.zeros([10]))

# initialize all variables into the session
# this must be done before variables can be used within a session
sess.run(tf.global_variables_initializer())

# TRAIN ========================================================================
# the regression model - prediction or classification
y = tf.matmul(x, W) + b

# finding the loss function
# using cross_entropy applied to the soft_max of target vs prediction
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
mean_cross_entropy = tf.reduce_mean(cross_entropy)

# using gradient descent to minimize cost
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(mean_cross_entropy)
# when train_step is run, it will:
#   apply gradient descent,
#   update parameters
# therefore, train the model by repeatedly running train_step

# we are running train 1000 times
for _ in range(1000):
  # the mini-batch of 100 samples for use in Stochastic Gradient Descent
  batch = mnist.train.next_batch(10)
  # feed_dict replaces the placeholder tensors x and y_ with
  # the training examples
  # note: can replace any tensor in graph with feed_dict
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# EVALUATE =====================================================================
# tf.argmax finds the index of the highest value along an axis
# in this case, the index corresponds to the class (because its one-hot)
# and the axis is 1 because we are one-dimentional here
predicted_class = tf.argmax(y, 1)
actual_class = tf.argmax(y_, 1)

# below returns an array of booleans [True, False...]
correct_prediction = tf.equal(predicted_class, actual_class)
# we want to cast those to numbers and calculate the percentage True
# the percentage True is just the average of the array
# [1,1,0,1] = 0.75 = 75%
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# I don't know how this .eval works.... I guess it evalates some shit....
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
