import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math

from ModelScripts import ModelSaver, ModelRestorer

fileDir = os.path.dirname(__file__)


def getData():
    dataDir = os.path.join(os.path.dirname(fileDir), 'Data')
    train_data = pd.read_csv(os.path.join(dataDir,'titanic_train.csv'))
    test_data = pd.read_csv(os.path.join(dataDir, 'titanic_test.csv'))

    class_list = train_data['Pclass'].fillna(0).tolist()
    sex_list = train_data['Sex'].fillna(0).tolist()
    age_list = train_data['Age'].fillna(0).tolist()
    sibling_parent_list = train_data['SibSp'].fillna(0).tolist()
    parent_child_list = train_data['Parch'].fillna(0).tolist()
    cabin_list = train_data['Cabin'].fillna(0).tolist()
    survived_list = train_data['Survived'].fillna(0).tolist()

    for person in range(len(sex_list)):

        if sex_list[person] == 'female':
            sex_list[person] = 0
        else:
            sex_list[person] = 1

        if cabin_list[person]:
            cabin_list[person] = 1
        else:
            cabin_list[person] = 0

    train_input = np.array([class_list,sex_list,age_list,sibling_parent_list,parent_child_list,cabin_list])
    train_output = np.array([survived_list])

    class_list = test_data['Pclass'].fillna(0).tolist()
    sex_list = test_data['Sex'].fillna(0).tolist()
    age_list = test_data['Age'].fillna(0).tolist()
    sibling_parent_list = test_data['SibSp'].fillna(0).tolist()
    parent_child_list = test_data['Parch'].fillna(0).tolist()
    cabin_list = test_data['Cabin'].fillna(0).tolist()
    survived_list = test_data['Survived'].fillna(0).tolist()

    for person in range(len(sex_list)):

        if sex_list[person] == 'female':
            sex_list[person] = 0
        else:
            sex_list[person] = 1

        if math.isnan(age_list[person]):
            age_list[person] = 0

        if cabin_list[person]:
            cabin_list[person] = 1
        else:
            cabin_list[person] = 0

    test_input = np.array([class_list, sex_list, age_list, sibling_parent_list, parent_child_list, cabin_list])
    test_output = np.array([survived_list])

    return train_input,train_output,test_input,test_output


def Model():
    # get the data
    X1, Y1, X2, Y2 = getData()

    Data = np.concatenate((np.concatenate((X1, X2), axis=1), np.concatenate((Y1, Y2), axis=1)), axis=0)
    np.random.shuffle(np.transpose(Data))
    (_, samples) = Data.shape
    X_data, Y_data = np.array_split(Data,[-1],axis=0)
    train_size = int(samples/2)
    remain = int(samples/2) + samples%2
    cross_size = int(remain/2)

    X_train, X_cross, X_test = np.array_split(X_data,[train_size, train_size+cross_size], axis=1)
    Y_train, Y_cross, Y_test = np.array_split(Y_data,[train_size, train_size+cross_size], axis=1)

    print(X_train.shape, X_cross.shape, X_test.shape, Y_train.shape, Y_cross.shape, Y_test.shape)
    (features, samples) = X_train.shape

    X = tf.placeholder(dtype=tf.float32, shape=[features, None], name="Input")
    Y = tf.placeholder(dtype=tf.float32, name="Output")

    W = tf.Variable(initial_value=np.random.rand(1,features), dtype=tf.float32, name="Weights")
    b = tf.Variable(initial_value=np.random.rand(), dtype=tf.float32, name="Bias")

    # define the hypothesis
    z = tf.add(tf.matmul(W, tf.nn.softmax(X)), b)
    pred = tf.nn.sigmoid(z, name="Prediction")

    # define the cost function
    cost_function_part_1 = tf.matmul(Y, tf.transpose(tf.log(pred)))[0]
    cost_function_part_2 = tf.matmul(1.0 - Y, tf.transpose(tf.log(1-pred)))[0]
    cost_function = tf.divide(tf.add(tf.multiply(-1.0,cost_function_part_1), tf.multiply(-1.0, cost_function_part_2))[0], samples)

    learningRate = [0.1, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]
    training_iterations = 100000
    display_step = 10000

    best_accuracy = -1
    best_rate = learningRate[0]
    for rate in learningRate:
        # define parameter learning algorithm
        optimizer = tf.train.GradientDescentOptimizer(rate).minimize(cost_function)

        # initialize all global variables
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for iteration in range(training_iterations):
                # Run optimization
                sess.run(optimizer, feed_dict={X: X_train, Y: Y_train})
                # Display logs each display_step
                if (iteration + 1)%display_step == 0:
                    cost = sess.run(cost_function, feed_dict={X: X_train, Y: Y_train})
                    print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(cost), "W=", sess.run(W),
                          "b=", sess.run(b))
            print("Optimization Finished!")
            training_cost = sess.run(cost_function, feed_dict={X: X_train, Y: Y_train})
            print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

            print("Cross Validating: ")
            output = sess.run(pred, feed_dict={X: X_cross})
            accuracy = tf.metrics.accuracy(labels=Y_cross, predictions=tf.cast(tf.round(output), tf.int32))
            sess.run(tf.local_variables_initializer())
            temp = sess.run(accuracy)[0] * 100

        if temp > best_accuracy:
            best_accuracy = temp
            best_rate = rate
        print("Accuracy: ",best_accuracy)
        print("Rate: ",best_rate)

    # define model name for storing
    model = os.path.basename(__file__).split('.')[0]
    cost_history = np.empty(0, dtype=float)

    # define parameter learning algorithm
    optimizer = tf.train.GradientDescentOptimizer(best_rate).minimize(cost_function)

    # initialize all global variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for iteration in range(training_iterations):
            # Run optimization
            sess.run(optimizer, feed_dict={X: X_train, Y: Y_train})
            # Display logs each display_step
            if (iteration + 1) % display_step == 0:
                cost = sess.run(cost_function, feed_dict={X: X_train, Y: Y_train})
                cost_history = np.append(cost_history, cost)
                print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(cost), "W=", sess.run(W),
                      "b=", sess.run(b))
        print("Optimization Finished!")
        training_cost = sess.run(cost_function, feed_dict={X: X_train, Y: Y_train})
        print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

        # Save the model for future use
        ModelSaver.save(filePath=fileDir, modelName=model, session=sess)

        print("Testing Model: ")
        output = sess.run(pred, feed_dict={X: X_test})
        accuracy = tf.metrics.accuracy(labels=Y_test, predictions=tf.cast(tf.round(output), tf.int32))
        sess.run(tf.local_variables_initializer())
        print("Accuracy: ", sess.run(accuracy)[0] * 100)

    print("Best learning rate: ",best_rate)
    plt.plot(cost_history)
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.axis([0, training_iterations/display_step, 0, np.max(cost_history)])
    plt.show()
    return


if __name__ == '__main__':
    Model()
