import tensorflow as tf

save_file = 'model.ckpt'

weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

# Class used to save and restore Tensor Variables
saver = tf.train.Saver()

with tf.Session() as sess:
	#initialize all the Variables
	sess.run(tf.global_variables_initializer())

	#showing the values of weights and biases
	print('Weights are:')
	print(sess.run(weights))
	print('Biases are:')
	print(sess.run(bias))

	# Save the model
	saver.save(sess, save_file)