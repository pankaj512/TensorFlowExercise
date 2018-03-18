import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

# First, load the image again
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/MarshOrchid.jpg"

image = mpimg.imread(filename)
height, width, depth = image.shape

# Create a TensorFlow Variable
x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    #x = tf.transpose(x, perm=[1, 0, 2])
    #x = tf.reverse_sequence(x, [height] * width, seq_dim=1, batch_dim=0)
    x = tf.reverse(x, axis=[1])
    session.run(model)
    result = session.run(x)

print(result.shape)
plt.imshow(result)
plt.show()