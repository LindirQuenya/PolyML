#!/usr/bin/env python3

# Non-builtins
import torchvision
import numpy as np

# Builtins
from pathlib import Path
import pickle
import gzip

# Change this to wherever you want the EMNIST DB to be downloaded.
parentDir = Path(__file__).parent.absolute()

# Warning, it's a bit over 2 GB.
print("Warning: this will download a bit over 2GB of data to {0}/EMNIST/raw/".format(parentDir))
print("Continuing.") if input("Continue? (y/n): ").lower() == 'y' else exit(0)

# This downloads the DB (if not already downloaded) and loads it as torch_emnist.
torch_emnist_badform = torchvision.datasets.EMNIST(root=parentDir, split='digits', download=True)

# It's in the wrong format. I want it as an np array, so that I can shuffle it.
torch_emnist = np.array(torch_emnist_badform, dtype='object')

# Shuffle it around, so that we're not holding on to any biases from data order.
np.random.shuffle(torch_emnist)

# A bit complicated (worse than complex :P), so let me try to explain.
# torch_emnist is an array of image-value pairs (tuples). The images are PIL, and the values are ints.
# The MNIST training data pickle is a tuple, of which the first element is an np array containing the images,
# and the second element is an np array containing the expected output values.
# Specifically, the first element of each MNIST tuple has shape (n, 784) => each image is formatted as a row vector.

# We need to go from pytorch format to MNIST pickle format.
# torch_emnist[i][0] contains the ith PIL image. We convert that to an np array, and divide by 255.
# This fixes the scaling (0-255) -> (0-1). Division is expensive, but multiplication by reciprocal is cheap.
# We flatten it to get it to a single row vector. Then we manually set the type to float32, in case it didn't
# get the hint from the division. Also, reciprocal is done with 1.0 so that cavemen won't have python2 troubles.
# Then we repeat this for every image in torch_emnist. Finally, we convert the whole thing to an np array.
arr_images = np.array([(np.array(torch_emnist[i][0])*(1.0/255)).flatten().astype('float32') for i in range(len(torch_emnist))])

# Similar idea: we get all the expected values (ints) from torch_emnist, put them in one giant np array.
arr_vals = np.array([torch_emnist[i][1] for i in range(len(torch_emnist))])

# Now we construct the three tuples. One is training, one is validation, and one is testing.
training_data = (arr_images[:-20000], arr_vals[:-20000])
validation_data = (arr_images[-20000:-10000], arr_vals[-20000:-10000])
test_data = (arr_images[-10000:], arr_vals[-10000:])

# Pickling time!
with gzip.open(parentDir/'emnist.pkl.gz', 'w') as f:
    p = pickle.Pickler(f)
    p.dump((training_data, validation_data, test_data))
