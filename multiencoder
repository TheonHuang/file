from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=False)

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
X = tf.placeholder("float", [None, n_input])
x_image = tf.reshape(X, [-1,28,28,1])

#cnn preparation
def weight_variable(shape):
     initial = tf.truncated_normal(shape,stddev=0.1)
     return tf.Variable(initial)
 
def bias_variable(shape):
     initial = tf.random_normal(shape,stddev=0.1)
     return tf.Variable(initial)
 
def conv_2d(x,w):
    return tf.nn.conv2d(x,w,strides = [1,1,1,1],padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
#build up autoencoder with cnn structure 
def encoder1(x):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv_2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv_2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    w_conv3 = weight_variable([4,4,64,128])
    b_conv3 = bias_variable([128])
    h_out = tf.nn.relu(tf.nn.conv2d(h_pool2,w_conv3,strides = [1,1,1,1],padding = 'VALID')+b_conv3)
    return h_out

def decoder1(x):
    w_fc1 = weight_variable([4 * 4 * 128,128])
    b_fc1 = bias_variable([128])
    pool2_flat = tf.reshape(x,[-1,4*4*128])
    hy_fc = tf.nn.relu(tf.matmul(pool2_flat,w_fc1)+b_fc1)
    w_fc2 = weight_variable([128,256])
    b_fc2 = bias_variable([256])
    y_2 = tf.matmul(hy_fc,w_fc2)+b_fc2
    w_out = weight_variable([256,784])
    b_out = bias_variable([784])
    y_out = tf.matmul(y_2,w_out)+b_out
    return y_out
    

print("helo")
encoder_op = encoder1(x_image)
decoder_op = decoder1(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X
# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
print("ca")
# Launch the graph
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)   
    for epoch in range(training_epochs):        
        # Loop over all batches
        for i in range(50):
            print(i)
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")
    # # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    plt.show()
    
