import tensorflow as tf

#name of the ckpt file that is going to be imported
save_file = 'model.ckpt'

#Remove the previous weghts and bias
tf.reset_default_graph()

#Two Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

#class used to save and restore tensor variables
saver = tf.train.Saver()

with tf.Session() as sess:
	#Load the weights and bias
	saver.restore(sess, save_file)

	#Show the values of inported weights and biases
	print("Weights : ")
	print(sess.run(weights))
	print("Bias : ")
	print(sess.run(bias))