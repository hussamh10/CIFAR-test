
# coding: utf-8

# In[3]:


import pickle
import numpy as np
import tensorflow as tf
file = 'data_batch_1'


# CIFAR-10  data is loaded, only one batch is being used right now.

# In[11]:


with open(file, 'rb') as f:
    dict = pickle.load(f, encoding='bytes')
images = dict[b'data']
labels = dict[b'labels']
labels = np.array(labels)
labels = np.reshape(labels, (10000, 1))


# Constants and parameters are set up

# In[12]:


x = tf.placeholder('float', [10000, 3072])
y = tf.placeholder('float', [10000, 1])


# Functions for actually performing convolutions and pooling

# In[13]:


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# In[14]:


def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,3,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([8*8*64, 1024])),
               'out':tf.Variable(tf.random_normal([1024, 1]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([1]))}

    x = tf.reshape(x, shape=[-1, 32, 32, 3])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)
    fc = tf.reshape(conv2 ,[-1, 8*8*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output


# In[15]:


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    print(prediction.shape, y.shape)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            _, c = sess.run([optimizer, cost], feed_dict={x:images, y:labels})
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(prediction, y)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


# In[16]:


train_neural_network(x)

