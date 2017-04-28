# Template for project/homework, learning with deep convolutional network
# The template program starts with a 2x2 max-pool to reduce input size.
# (This is a very bad idea for anything but academic experiments...)
# The template shows how to use a convolutional layer, a fully connected
# layer, and dropout

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(X, W, padding):
    return tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding=padding)


def max_pool_2x2(X, padding):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding=padding)

# print(X.shape)

orig_image = tf.reshape(X, [-1, 28, 28, 1])
### For this assignment 2x2 max pool layer must be the first layer ###
h_pool0 = max_pool_2x2(orig_image, 'SAME')
print("Pool 0:" + str(h_pool0.shape))
### End of first max pool layer ###

# beginning of layer definitions

# Convolutional Layer #1

W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1, 'SAME') + b_conv1)
print("Conv 1:" + str(h_conv1.shape))

# Max-Pool Layer #1

h_pool1 = max_pool_2x2(h_conv1, 'VALID')
print("Pool 2:" + str(h_pool1.shape))

# Convolutional Layer #2

W_conv2_1 = weight_variable([3, 3, 32, 64])
b_conv2_1 = bias_variable([64])

h_conv2_1 = tf.nn.relu(conv2d(h_pool1, W_conv2_1, 'SAME') + b_conv2_1)
print("Conv 2_1:" + str(h_conv2_1.shape))

W_conv2_2 = weight_variable([1, 1, 32, 64])
b_conv2_2 = bias_variable([64])

h_conv2_2 = tf.nn.relu(conv2d(h_pool1, W_conv2_2, 'SAME') + b_conv2_2)
print("Conv 2_2:" + str(h_conv2_2.shape))

W_conv2_3 = weight_variable([5, 5, 32, 64])
b_conv2_3 = bias_variable([64])

h_conv2_3 = tf.nn.relu(conv2d(h_pool1, W_conv2_3, 'SAME') + b_conv2_3)
print("Conv 2_3:" + str(h_conv2_3.shape))

h_conv2 = (h_conv2_1 + h_conv2_2 + h_conv2_3)
print("Conv 2:" + str(h_conv2.shape))

# Convolutional Layer #3

W_conv3_1 = weight_variable([3, 3, 64, 128])
b_conv3_1 = bias_variable([128])

h_conv3_1 = tf.nn.relu(conv2d(h_conv2, W_conv3_1, 'SAME') + b_conv3_1)
print("Conv 3_1:" + str(h_conv3_1.shape))


W_conv3_1b = weight_variable([1, 1, 128, 128])
b_conv3_1b = bias_variable([128])

h_conv3_1b = tf.nn.relu(conv2d(h_conv3_1, W_conv3_1b, 'SAME') + b_conv3_1b)
print("Conv 3_2:" + str(h_conv3_1b.shape))

W_conv3_3 = weight_variable([5, 5, 64, 128])
b_conv3_3 = bias_variable([128])

h_conv3_3 = tf.nn.relu(conv2d(h_conv2, W_conv3_3, 'SAME') + b_conv3_3)
print("Conv 3_3:" + str(h_conv3_3.shape))

h_conv3 = (h_conv3_1b + h_conv3_3)
print("Conv 3:" + str(h_conv3.shape))

# Convolutional Layer #4

# W_conv4_1 = weight_variable([3, 3, 128, 256])
# b_conv4_1 = bias_variable([256])

# h_conv4_1 = tf.nn.relu(conv2d(h_conv3, W_conv4_1, 'SAME') + b_conv4_1)
# print("Conv 4_1:" + str(h_conv4_1.shape))

# W_conv4_2 = weight_variable([5, 5, 128, 256])
# b_conv4_2 = bias_variable([256])

# h_conv4_2 = tf.nn.relu(conv2d(h_conv3, W_conv4_2, 'SAME') + b_conv4_2)
# print("Conv 4_2:" + str(h_conv4_2.shape))


# h_conv4 = (h_conv4_1 + h_conv4_2)
# print("Conv 4:" + str(h_conv4.shape))

# Densely Connected Layer

W_fc1 = weight_variable([7 * 7 * 128, 1024])
b_fc1 = bias_variable([1024])

h_pool4_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
print("FC 6:" + str(h_fc1.shape))

# Dropout

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# end of layer definitions

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_conv))
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            X: batch[0], Y: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={X: batch[0], Y: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={
    X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}))
