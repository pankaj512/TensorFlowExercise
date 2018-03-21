import os
import tensorflow as tf


def getModel(modelName, session):

    # get model directory
    fileDir = os.path.dirname(os.path.realpath(__file__))

    # get model name
    model = os.path.join(os.path.dirname(fileDir), 'Models', modelName, modelName)

    saver = tf.train.import_meta_graph(model + '.meta')
    # create model saver/restorer
    saver = tf.train.Saver()

    # restore model
    saver.restore(session,model)
    return session
