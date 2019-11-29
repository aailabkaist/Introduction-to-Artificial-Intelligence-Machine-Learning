
# coding: utf-8

# ## Single Layer Neural Network with MNIST data

# In[4]:



import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#tensorflow is only used for loading mnist data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


# ## Model Setting

# In[5]:


# N = width (# of nodes) of hidden layer
n = 100

# batch & epoch
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)
epoch_size = 80

# activation function definition (logistic function)
def Logis(x):
    y = 1/(1+np.exp(-x))
    return y


# ## Weight & Bias term initialization

# In[7]:


'''weight & biased term initialization (xavier initializer)'''
w1 = np.random.uniform(-np.sqrt(6.0/(784+n)),np.sqrt(6.0/(784+n)),(784,n))
b1 = np.random.uniform(-np.sqrt(6.0/(784+n)), np.sqrt(6.0/(784+n)),(1,n))

w2 = np.random.uniform(-np.sqrt(6.0/(n+10)),np.sqrt(6.0/(n+10)),(n,10))
b2 = np.random.uniform(-np.sqrt(6.0/(n+10)),np.sqrt(6.0/(n+10)),(1,10))


# ## BackPropagation 

# In[8]:


for epoch in range(epoch_size):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        X = batch_xs
        Y = batch_ys
        '''forward-pass of neural network'''
        #input activation for Single hidden layer
        L1 = np.matmul(X,w1)+b1
        #output activation for Single hidden layer (function : logistic function)
        L1 = Logis(L1)

        # input activation for Output layer
        L2 = np.matmul(L1,w2)+b2
        # output activation for Output layer (function : softmax function)
        Y_hat = np.exp(L2)/np.exp(L2).sum(axis=1).reshape(-1,1)

        '''backward-pass of neural network'''
        #delta signal1 defined
        delt_1 = np.dot(np.dot(Y_hat,np.ones((10,batch_size))-Y_hat.T),Y-Y_hat)
        #gradient of w2, b2
        w2_gradient = (-np.dot(L1.T,delt_1))/batch_size
        b2_gradient = (-np.average(delt_1, axis=0))

        #delta signal2 defined
        delt_2 = np.dot(delt_1,w2.T)*L1*(1-L1)
        #gradient of w1, b1
        w1_gradient = (-np.dot(X.T,delt_2))/batch_size
        b1_gradient = (-np.average(delt_2, axis=0))

        '''weight update'''
        learning_rate = 0.001
        w2 -= learning_rate*w2_gradient
        b2 -= learning_rate*b2_gradient
        w1 -= learning_rate*w1_gradient
        b1 -= learning_rate*b1_gradient

        total_cost += np.mean(0.5*np.square(Y-Y_hat))

    #print total_cost
    total_cost = total_cost/batch_size
    if epoch % 10 == 9:
        print('Epoch : ', '%04d' % (epoch+1), 'cost =', '%.5f' % (total_cost))

print('optimization complete')


# ## Accuracy Test (Classifier)

# In[9]:


'''classification accuracy test'''
X_test = mnist.test.images
Y_test = mnist.test.labels
L1_test = np.matmul(X_test,w1)+b1
L1_test = Logis(L1_test)
L2_test = np.matmul(L1_test,w2)+b2
Y_hat_test = np.exp(L2_test)/np.exp(L2_test).sum(axis=1).reshape(-1,1)

is_correct = np.equal(np.argmax(Y_hat_test,1),np.argmax(Y_test,1))
accuracy = np.sum(is_correct.astype(int))*0.0001

print ('accuracy is :','%.5f' % (accuracy))

