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


def createModel():
    # get the data
    X_train, Y_train, X_test, Y_test = getData()
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    (features, samples) = X_train.shape

    X = tf.placeholder(dtype=tf.float32, shape=[features, None], name="Input")
    Y = tf.placeholder(dtype=tf.float32, name="Output")

    W = tf.Variable(initial_value=np.random.rand(1,features), dtype=tf.float32, name="Weights")
    b = tf.Variable(initial_value=np.random.rand(), dtype=tf.float32, name="Bias")

    # define the hypothesis
    pred = tf.add(tf.matmul(W, X), b, name="Prediction")

    # define the cost function
    cost_function = tf.losses.mean_squared_error(Y, pred)

    # define parameter learning algorithm
    learningRate = 0.0001
    optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_function)

    # initialize all global variables
    init = tf.global_variables_initializer()

    # define model name for storing
    model = os.path.basename(__file__).split('.')[0]

    training_iterations = 10000
    display_step = 50
    cost_history = np.empty(0, dtype=float)

    with tf.Session() as sess:
        sess.run(init)
        for iteration in range(training_iterations):
            # Run optimization
            sess.run(optimizer, feed_dict={X: X_train, Y: Y_train})
            # Display logs each display_step
            if (iteration + 1)%display_step == 0:
                cost = sess.run(cost_function, feed_dict={X: X_train, Y: Y_train})
                cost_history = np.append(cost_history, cost)
                print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(cost), "W=", sess.run(W), "b=",
                      sess.run(b))
        print("Optimization Finished!")
        training_cost = sess.run(cost_function, feed_dict={X: X_train, Y: Y_train})
        print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

        # Save the model for future use
        ModelSaver.save(filePath=fileDir, modelName=model, session=sess)

    plt.plot(cost_history)
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.axis([0, training_iterations/display_step, 0, np.max(cost_history)])
    plt.show()
    return


def testModel():
    X_train, Y_train, X_test, Y_test = getData()
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    (features, samples) = X_train.shape

    X = tf.placeholder(dtype=tf.float32, shape=[features, None])
    Y = tf.placeholder(dtype=tf.float32)

    W = tf.Variable(initial_value=np.random.rand(1, features), dtype=tf.float32)
    b = tf.Variable(initial_value=np.random.rand(), dtype=tf.float32)

    init = tf.global_variables_initializer()

    # define model name for restoring
    model = os.path.basename(__file__).split('.')[0]

    with tf.Session() as sess:
        sess.run(init)
        print("Restoring the model\n")
        ModelRestorer.getModel(sess, fileDir)
        input = sess.get_operation_by_name("Input").outputs[0]
        prediction = sess.get_operation_by_name("Prediction").outputs[0]

    return


if __name__ == '__main__':
    createModel()
    #testModel()
