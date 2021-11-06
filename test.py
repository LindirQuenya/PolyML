#!/usr/bin/env python3
import network
import mnist_loader

polynet = network.Network([782, 32, 32, 10], [1, 2, 3])
linnet = network.Network([782, 32, 32, 10])
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

polynet.SGD(training_data, 120, 10, 3.0)
linnet.SGD(training_data, 120, 10, 3.0)

print("Polynomial network: {0}/10000".format(polynet.evaluate(test_data)))
print("Linear network: {0}/10000".format(linnet.evaluate(test_data)))
