import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 1000

n_nodes_hl4 = 900
n_nodes_hl5 = 900
n_nodes_hl6 = 900

n_nodes_hl7 = 800
n_nodes_hl8 = 800
n_nodes_hl9 = 800

n_nodes_hl10 = 500
n_nodes_hl11 = 500
n_nodes_hl12 = 500




n_classes = 3
batch_size = 10

x = tf.placeholder('float', [None, 2500])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([2500, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    hidden_4_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))}

    hidden_5_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl5]))}

    hidden_6_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl5, n_nodes_hl6])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl6]))}

    hidden_7_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl6, n_nodes_hl7])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl7]))}

    hidden_8_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl7, n_nodes_hl8])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl8]))}

    hidden_9_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl8, n_nodes_hl9])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl9]))}

    hidden_10_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl9, n_nodes_hl10])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl10]))}

    hidden_11_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl10, n_nodes_hl11])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl11]))}

    hidden_12_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl11, n_nodes_hl12])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl12]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl12, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes])), }

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.relu(l4)

    l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['biases'])
    l5 = tf.nn.relu(l5)

    l6 = tf.add(tf.matmul(l5, hidden_6_layer['weights']), hidden_6_layer['biases'])
    l6 = tf.nn.relu(l6)

    l7 = tf.add(tf.matmul(l6, hidden_7_layer['weights']), hidden_7_layer['biases'])
    l7 = tf.nn.relu(l7)

    l8 = tf.add(tf.matmul(l7, hidden_8_layer['weights']), hidden_8_layer['biases'])
    l8 = tf.nn.relu(l8)

    l9 = tf.add(tf.matmul(l8, hidden_9_layer['weights']), hidden_9_layer['biases'])
    l9 = tf.nn.relu(l9)

    l10 = tf.add(tf.matmul(l9, hidden_10_layer['weights']), hidden_10_layer['biases'])
    l10 = tf.nn.relu(l10)

    l11 = tf.add(tf.matmul(l10, hidden_11_layer['weights']), hidden_11_layer['biases'])
    l11 = tf.nn.relu(l11)

    l12 = tf.add(tf.matmul(l11, hidden_12_layer['weights']), hidden_12_layer['biases'])
    l12 = tf.nn.relu(l12)

    output = tf.matmul(l12, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        # sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)