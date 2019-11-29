import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


class NVDM:

    network_architecture = []
    transfer_fct = 0

    x = []
    x_reconstr_mean = []
    z = []
    z_mean = []
    z_log_sigma_sq = []

    sess = 0
    cost = 0
    optimizer = 0

    def __init__(self,network_architecture,transfer_fct=tf.nn.relu):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct

        self.x = tf.placeholder(tf.float32,[None,network_architecture["n_input"]])

    def _create_network(self,batch_size):
        # initialize weight -> recognition network -> Reparameterization trick -> generator network
        
        network_weights = self._initialize_weights(**self.network_architecture)

        self.z_mean, self.z_log_sigma_sq = self._recognition_network(network_weights["weights_recog"],network_weights["biases_recog"])

        n_z = self.network_architecture["n_z"]
        
        # Reparameteriation trick
        eps = tf.random_normal((batch_size,n_z),0,1,dtype=tf.float32)

        self.z = tf.add(self.z_mean,tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)),eps))

        self.pb_x = self._generator_network(network_weights["weights_gener"],network_weights["biases_gener"])

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, n_input, n_z):
        # Input : n_hidden_recog_1, n_hidden_recog_2, n_input, n_z
        # Output : all_weights
        
        all_weights = dict()

        all_weights['weights_recog'] = { 'h1' : tf.Variable(self.xavier_init(n_input,n_hidden_recog_1)),                                          'h2' : tf.Variable(self.xavier_init(n_hidden_recog_1,n_hidden_recog_2)),                                          'out_mean': tf.Variable(self.xavier_init(n_hidden_recog_2, n_z)),                                          'out_log_sigma': tf.Variable(self.xavier_init(n_hidden_recog_2, n_z))}

        all_weights['biases_recog'] = { 'b1' : tf.Variable(tf.zeros([n_hidden_recog_1],dtype=tf.float32)),                                         'b2' : tf.Variable(tf.zeros([n_hidden_recog_2],dtype=tf.float32)),                                         'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),                                         'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}

        all_weights['weights_gener'] = {'R': tf.Variable(self.xavier_init(n_z, n_input))}

        all_weights['biases_gener'] = {'b': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights

    def _recognition_network(self,weights,biases):
        # VAE recognition network method
        # Input : (recognition network) weights, biases
        # Output: (z_mean, z_log_sigma_sq)
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), biases['b1'])) # L1 = g(b1 + W1*X)
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])) # L2 = g(b2 + W2*L1)
        z_mean = tf.add(tf.matmul(layer_2,weights['out_mean']),biases['out_mean']) # z_mean = b_{out_mean} + W_{out mean}*L2
        z_log_sigma_sq = tf.add(tf.matmul(layer_2,weights['out_log_sigma']),biases['out_log_sigma'])
        # z_log_sigma_sq = b_{log_sigma_sq} + W_{log_sigma_sq}*L2
        return (z_mean,z_log_sigma_sq)

    def _generator_network(self,weights,biases):
        # VAE generator network method
        # Input : (generator network) weights, biases
        # Output : x_reconstr_mean
        pb_x = tf.nn.softmax(tf.add(tf.matmul(self.z, weights['R']), biases['b'])) # L1 = g(b1 + W1*Z)
        return pb_x

    def _create_loss_optimizer(self,learning_rate):
        # VAE total cost minimize optimizer
        # Input : learning_rate
        # Output : None
        
        # reconstruction loss function = binary cross-entropy function
        reconstr_loss = -tf.reduce_sum(tf.multiply(tf.log(self.pb_x),self.x),1)
        
        # latent loss function = KL-divergence
        latent_loss = -0.5*tf.reduce_sum(1+self.z_log_sigma_sq-tf.square(self.z_mean)-tf.exp(self.z_log_sigma_sq),1)
        
        # (VAE total) cost = reconstruction loss + latent loss
        self.cost = tf.reduce_mean(reconstr_loss+latent_loss)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

    def partial_fit(self,x):
        # VAE Input data x cost
        # Input : x
        # Output : cost
        opt,cost = self.sess.run((self.optimizer,self.cost),feed_dict={self.x:x})
        return cost

    def transform(self,x):
        # VAE encoder network latent variable z mean return
        # Input : x
        # Output : z_mean
        return self.sess.run(self.z_mean,feed_dict={self.x:x})

    def generate(self,z_mu=None):
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        return self.sess.run(self.x_reconstr_mean,feed_dict={self.z:z_mu})

    def reconstruct(self,x):
        # VAE X reconstruct mean return
        # Input : x
        # Output : (VAE) x_reconstr_mean
        return self.sess.run(self.x_reconstr_mean,feed_dict={self.x:x})

    def xavier_init(self,fan_in, fan_out, constant=1): 
        # Xavier initialization method
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))

        return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

    def train(self,trainX,batch_size=100,training_epochs=500,learning_rate=0.0005):
        # Input : trainX, batch_size, training_epochs, learning_rate
        # Output : None

        total_costs = np.zeros(training_epochs)
        total_perplexity = np.zeros(training_epochs)

        self._create_network(batch_size) # VAE
        self._create_loss_optimizer(learning_rate) # VAE loss function optimizer

        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # session : tensorflow
        self.sess.run(init)

        startTime = time.time()
        for epoch in range(training_epochs): # training_epochs
            avg_cost = 0. # training set average cost
            avg_perplexity = 0.
            total_batch = int(len(trainX)/batch_size) # total batch : batch

            for i in range(total_batch): # total_batch
                batch = []
                for j in range(batch_size):
                    batch.append(trainX[i*batch_size+j]) # training X batch

                cost = self.partial_fit(batch) # i batch cost
                avg_cost += cost / total_batch
                avg_perplexity += cost / (total_batch * (sum(map(sum,batch))/batch_size))

            total_costs[epoch] = avg_cost
            total_perplexity[epoch] = np.exp(avg_perplexity)

            print("Epoch : ",'%04d'%(epoch+1)," Cost = ","{:.9f}".format(avg_cost)," perplexity = ","{:.9f}".format(np.exp(avg_perplexity)))
            print("Elapsed Time : " + str((time.time() - startTime)))

        plt.plot(total_costs)
        plt.xlabel('epoch')
        plt.ylabel('cost')
        plt.show()
        return


# In[3]:

print("start!")
newsgroups_train = fetch_20newsgroups(subset='train',shuffle=True, random_state=0) # load "train" dataset

bagVocab = np.load('bagVocab_3+_stop_tfidf.npy')
target = newsgroups_train.target  # topic of each news

vect = CountVectorizer(vocabulary=bagVocab, binary=True)  # the way of count words (in bagVocab)
data = vect.fit_transform(newsgroups_train.data)  # count words in each news (in bagVocab)

newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=0)
data_test = vect.fit_transform(newsgroups_test.data)
target_test = newsgroups_test.target

print("end!")

trainX = []

for i in range(data.shape[0]):
    tempdata = data[i].toarray()
    temptuple = tuple(map(tuple,tempdata))[0]
    trainX.append(temptuple)

np.random.seed(0)
tf.set_random_seed(0)


# 784 500 500 10 500 500 784
network_architecture = dict(n_hidden_recog_1=500, n_hidden_recog_2=500, n_input=data.shape[1], n_z=50)

nvdm = NVDM(network_architecture)
nvdm.train(trainX,batch_size=100,training_epochs=100,learning_rate=0.001) # VAE training