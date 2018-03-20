import os
import tensorflow as tf


def save(filePath, modelName, session):

    # get model directory and Name
    finalDir = os.path.join(os.path.dirname(filePath), 'Models', modelName)
    if not os.path.exists(finalDir):
        os.makedirs(finalDir)
    model = os.path.join(finalDir, modelName)

    # get the tesnor model saver
    saver = tf.train.Saver()

    # Save the model for future use
    saver.save(session, model)

    return
