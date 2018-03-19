import os

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

# My Import
from ModelScripts import ModelSaver

fileDir = os.path.dirname(os.path.realpath(__file__))

def createModel():
    # create data
    x_train = np.arange(10)

    y_train = x_train * 2 + 1

    X = tf.placeholder(dtype=tf.float32)
    Y = tf.placeholder(dtype=tf.float32)

    W = tf.Variable(initial_value=np.random.rand(), dtype=tf.float32)
    b = tf.Variable(initial_value=np.random.rand(), dtype=tf.float32)

    # define the hypothesis
    pred = np.dot(X,W)+b

    # define the cost function
    cost_function = tf.losses.mean_squared_error(Y, pred)

    learningRate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_function)

    init = tf.global_variables_initializer()

    model = os.path.basename(__file__).split('.')[0]

    training_epochs = 1000
    display_step = 50
    cost_history = np.empty(0, dtype=float)

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            # Run optimization
            sess.run(optimizer, feed_dict={X: x_train, Y: y_train})
            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                cost = sess.run(cost_function, feed_dict={X: x_train, Y: y_train})
                cost_history = np.append(cost_history, cost)
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost), "W=", sess.run(W), "b=", sess.run(b))

        print("Optimization Finished!")
        training_cost = sess.run(cost_function, feed_dict={X: x_train, Y: y_train})
        print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

        # Graphic display
        plt.plot(x_train, y_train, 'ro', label='Original data')
        plt.plot(x_train, sess.run(W) * x_train + sess.run(b), label='Fitted line')
        plt.legend()
        plt.show()

        # Save the model for future use
        ModelSaver.save(filePath=fileDir, modelName=model, session=sess)

    plt.plot(cost_history)
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.axis([0, 50, 0, np.max(cost_history)])
    plt.show()


def testModel():

    return


if __name__ == '__main__':
    createModel()
    testModel()