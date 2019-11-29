





from tensorflow.examples.tutorials.mnist import input_data

    self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    batch_x, batch_y = self.mnist.train.next_batch(Batch_size)
    test_data = self.mnist.test.images[:test_lne].reshape((-1 self.n_steps, self.n_input))
    test_label = self.mnist.test.labels[:test_len]



def setupRNN(self):
    x = tf.unstack(self.x, self.n_steps, 1)
    lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    weights = {'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([self.n_classes]))}
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def trainRNN(self, training, iters, batch_size):
    init = tf.global_variables_initializer()
    self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    self.sess = tf.Session()
    self.sess.run(init)
    step=0

    while step < training_iter:
        batch_x, batch_y = self.mnist.train.next_batch(batch_size)

        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, self.n_steps, self.n_inputs))
        self.sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})
        acc = self.sess.run(self.accurach, feed_dict={self.x: batch_x, self.y: batch_y})
        loss = self.sess.run(self.cost, feed_dict={self.x: batch_x, self.y: batch_y})
        print("Iter " + str(step) +
              ", Minibatch Loss= " + "{:.6f}".format(loss) +
              ", Training Accurach= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

def setupLoss(self.learning_rate):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)
    return optimizer, cost
