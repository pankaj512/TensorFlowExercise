import matplotlib.image as mpimg
import os
# First, load the image
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
filename = dir_path + "/MarshOrchid.jpg"

# Load the image
image = mpimg.imread(filename)

# Print out its shape
print(image.shape)