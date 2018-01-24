import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

#remove previous tensors and operations
tf.reset_default_graph()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('.', one_hot="True")

learning_rate = 0.001
n_input = 784
n_classes = 10

#features and labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

#Weights and biases
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

#model
y = tf.add(tf.matmul(features, weights), bias)

#loss and optimzer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= y, labels= labels))
optimzer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Calculate accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#save file location and Saver object
save_file = 'train_model.ckpt'
saver = tf.train.Saver()


#training the model.
batch_size = 128
total_n_batches = mnist.train.num_examples / batch_size
n_epochs = 200
loss = []

#Launch the graph
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	#training cycle
	for epoch in range(n_epochs):
		for i in range(total_n_batches):
			batch_features, batch_labels = mnist.train.next_batch(batch_size)
			_, c = sess.run([optimzer, cost], feed_dict={features: batch_features, labels: batch_labels})
			loss.append(c)

		#Print status for each epoch
		epoch_accuracy = sess.run(accuracy, feed_dict={features: mnist.validation.images, labels: mnist.validation.labels})
		print("Epoch :", epoch, " validation accuracy: ", epoch_accuracy)

	#save the model (this save process should done within the session)
	saver.save(sess, save_file)
	print("Trained model saved")

plt.plot(loss)
plt.show()