import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fileDir = os.path.dirname(__file__)


def getData():
    dataDir = os.path.join(os.path.dirname(fileDir), 'Data')
    train_data = pd.read_csv(os.path.join(dataDir,'titanic_train.csv'))
    test_data = pd.read_csv(os.path.join(dataDir, 'titanic_test.csv'))

    class_list = train_data['PClass'].tolist()
    sex_list = train_data['Sex'].tolist()
    age_list = train_data['Age'].tolist()
    sibling_parent_list = train_data['SibSp']
    parent_child_list = train_data['Parch']
    cabin_list = train_data['Cabin']
    survived_list = train_data['Survived']

    for person in range(len(sex_list)):

        if sex_list[person] == 'female':
            sex_list[person] = 0
        else:
            sex_list[person] = 1

        if cabin_list[person]:
            cabin_list[person] = 1
        else:
            cabin_list[person] = 0

    train_input = np.array(class_list,sex_list,age_list,sibling_parent_list,parent_child_list,cabin_list)
    train_output = np.array(survived_list)

    class_list = test_data['PClass'].tolist()
    sex_list = test_data['Sex'].tolist()
    age_list = test_data['Age'].tolist()
    sibling_parent_list = test_data['SibSp']
    parent_child_list = test_data['Parch']
    cabin_list = test_data['Cabin']
    survived_list = test_data['Survived']

    for person in range(len(sex_list)):

        if sex_list[person] == 'female':
            sex_list[person] = 0
        else:
            sex_list[person] = 1

        if cabin_list[person]:
            cabin_list[person] = 1
        else:
            cabin_list[person] = 0

    test_input = np.array(class_list, sex_list, age_list, sibling_parent_list, parent_child_list, cabin_list)
    test_output = np.array(survived_list)

    return train_input,train_output,test_input,test_output


def createModel():
    X_train, Y_train, X_test, Y_test = getData()

    return


if __name__ == '__main__':
    createModel()
