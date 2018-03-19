import os
import tensorflow as tf


def save(filePath, modelName, session):

    # get model directory and Name
    finalDir = os.path.join(os.path.dirname(filePath), 'Models')
    os.makedirs(finalDir,mode=777,exist_ok=True)
    model = os.path.join(finalDir,modelName,modelName)

    # get the tesnor model saver
    saver = tf.train.Saver()

    # Save the model for future use
    saver.save(session, model)

    return
