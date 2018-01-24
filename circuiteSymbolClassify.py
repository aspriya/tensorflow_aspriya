from dataSetPreporationOldWay import get_data_set
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

train_x, train_y, test_x, test_y = get_data_set('tData/TEST/')


x = tf.placeholder(tf.float32, shape=[None, 2500])
y_ = tf.placeholder(tf.float32, shape=[None, 3])

h1_nodes = 500
h2_nodes = 500
h3_nodes = 400
output_nodes = 3
batch_size = 4

def define_model(data):
    hidden_layer1 = {
        'weights': tf.Variable(tf.random_normal([2500, h1_nodes])),
        'biases' : tf.Variable(tf.random_normal([h1_nodes]))
    }
    hidden_layer2 = {
        'weights' : tf.Variable(tf.random_normal([h1_nodes, h2_nodes])),
        'biases'  : tf.Variable(tf.random_normal([h2_nodes])) 
    }
    
    hidden_layer3 = {
        'weights' : tf.Variable(tf.random_normal([h2_nodes, h3_nodes])),
        'biases'  : tf.Variable(tf.random_normal([h3_nodes]))
    }
    output_layer = {
        'weights' : tf.Variable(tf.random_normal([h3_nodes, output_nodes])),
        'biases'  : tf.Variable(tf.random_normal([output_nodes]))
    }
    
    operation_h1 = tf.add(tf.matmul(data, hidden_layer1['weights']), hidden_layer1['biases'])
    operation_h1 = tf.nn.relu(operation_h1)
    
    operation_h2 = tf.add(tf.matmul(operation_h1, hidden_layer2['weights']), hidden_layer2['biases'])
    operation_h2 = tf.nn.relu(operation_h2)
    
    operation_h3 = tf.add(tf.matmul(operation_h2, hidden_layer3['weights']), hidden_layer3['biases'])
    operation_h3 = tf.nn.relu(operation_h3)
    
    output = tf.add(tf.matmul(operation_h3, output_layer['weights']), output_layer['biases'])
    
    return output

def training(features):
    prediction = define_model(features)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    loss_ = []
    
    num_epochs = 60
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
        
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, loss], feed_dict={x:batch_x, y_:batch_y})
                epoch_loss += c
                loss_.append(c)
                i += batch_size
            
            print('Epoch', epoch, ' completed out of ', num_epochs, ' loss: ', epoch_loss)
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))
        print('Accuracy : ', accuracy.eval({x:test_x, y_:test_y}))

    plt.plot(loss_)
    plt.show()

training(x)