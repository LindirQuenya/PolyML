import os
import pickle
import random
import numpy as np
from PIL import Image
from scipy.special import erf
#from nn.mnist_loader import load_data_wrapper



class Network(object):
    def __init__(self, sizes, polyorders=None, non_linearity = "sigmoid"):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network. The list ``polyorders``
        contains the order of the polynomials used in the connections
        between the layers. It must be one element shorter than ``sizes``.
        For example, if ``sizes`` was [2, 3, 1] and ``polyorders`` was
        [4, 12] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron. The connections between the first
        and second layer would be fourth-order polynomials, and the
        connections between the second and third layer would be
        12th-order polynomials. If polyorders is not supplied, it is
        assumed to be all ones. The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.innums = self.sizes[:-1]
        # Set polyorders to all ones if it's unspecified.
        self.polyorders = polyorders if polyorders!=None else [1]*(len(sizes)-1)
        if np.isin(True, np.array(self.polyorders)<1):
            raise ValueError("Argument polyorders must have positive elements!")
        # Zero-order weights. Not worth making a matrix out of.
        self.biases = np.asarray([np.random.randn(y, 1) for y in sizes[1:]])
        # 1-n order weights. The zero-based indexing is deceptive here:
        # index zero corresponds to order 1, ind 1 -> ord 2, etc.
        self.weights = np.asarray([[np.random.randn(y, x) for i in range(p)]
                        for x, y, p in zip(sizes[:-1], sizes[1:], self.polyorders)])
        if non_linearity == "sigmoid":
            self.non_linearity = sigmoid
            self.d_non_linearity = sigmoid_prime
        elif non_linearity == "ReLU":
            self.non_linearity = ReLU
            self.d_non_linearity = ReLU_prime
        elif non_linearity == "GELU":
            self.non_linearity = GELU
            self.d_non_linearity = GELU_prime
        else:
            raise ValueError("Invalid non_linearity")

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, warr, sz in zip(self.biases, self.weights, self.innums):
            # This looks complicated, kinda is. For each power p = 1..k, it multiplies
            # np.power(a, p) by the corresponding weight matrix. Because of the zero-based
            # indexing, this ends up being n = 0..(k-1), np.power(a, n+1), and warr[n].
            # In the above, k is the polynomial order for that connection. Also (conveniently)
            # the length of warr for a given connection set.
            a = self.non_linearity(npadd_many(b,*[np.dot(warr[n], np.power(np.array(a),n+1))
                                                  for n in range(len(warr))]), sz)
        return a

    def get_activation_of_all_layers(self, in_a, n_layers = None, get_zs = False):
        """Gets the activations of the neurons and, if get_zs is True, their pre-non_linearity values."""
        if n_layers is None:
            n_layers = self.num_layers
        input_a = np.asarray(in_a)
        activations = [input_a.reshape((input_a.size, 1))]
        if get_zs:
            zs=[]
        for b, warr, sz in list(zip(self.biases, self.weights, self.innums))[:n_layers]:
            last_a = activations[-1]
            # See the comment in Network.feedforward() for an explaination of this.
            z = npadd_many(b,*[np.dot(warr[n], np.power(last_a,n+1)) for n in range(len(warr))])
            if get_zs:
                zs.append(z)
            new_a = self.non_linearity(z, sz)
            new_a = new_a.reshape((new_a.size, 1))
            activations.append(new_a)
        if get_zs:
            return np.array(activations), np.array(zs)
        else:
            return np.array(activations)

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = np.asarray([np.zeros(b.shape) for b in self.biases])
        nabla_w = np.asarray([[np.zeros(np.array(w).shape) for w in warr] for warr in self.weights])
        #nabla_b = np.zeros(self.biases.shape)
        #nabla_w = np.zeros(self.weights.shape)
        # feedforward
        activations, zs = self.get_activation_of_all_layers(x, get_zs=True)
        # backward pass
        # dC/dz2 = dC/dn2 * dn2/dz2
        delta = self.cost_derivative(activations[-1], y) * self.d_non_linearity(zs[-1], self.innums[-1])
        # dz2/dw2p0 = 1 -> dC/dw2p0 = dC/dz2
        nabla_b[-1] = delta
        # dz2/dw2pn = (n1)^n -> dC/dz2 dot (n1)^n
        #print(delta.shape, activations[-2].shape, nabla_w[-1].shape)
        nabla_w[-1] = [np.dot(delta, np.power(activations[-2], n+1).transpose()) for n in range(len(nabla_w[-1]))]
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            # dn1/dz1
            sp = self.d_non_linearity(z, self.innums[-l])
            # activation-weighting matrix used for dz/dnx. diagonal, for extra-special usefulness.
            actmat = np.diag(np.reshape(activations[-l], (len(activations[-l]))))
            # dz2/dn1
            dzdn1 = npadd_many(self.weights[-l+1][0],
                               *[np.multiply(n+1,np.matmul(self.weights[-l+1][n],np.power(actmat, n)))
                                 for n in range(1, len(self.weights[-l+1]))])
            #delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            #dC/dz1 = dC/dz2 * dz2/dn1 * dn1/dz1
            delta = np.dot(dzdn1.transpose(), delta) * sp
            nabla_b[-l] = delta
            #Um... I think?
            #print(delta.shape, activations[-l-1].shape)
            nabla_w[-l] = [np.dot(delta, np.power(activations[-l-1], n+1).transpose()) for n in range(len(nabla_w[-l]))]
        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = np.asarray([np.zeros(b.shape) for b in self.biases])
        nabla_w = np.asarray([np.zeros(np.array(w).shape) for w in self.weights])
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = np.asarray([[nw+dnw for nw, dnw in zip(nwarr, dnwarr)]
                       for nwarr, dnwarr in zip(nabla_w, delta_nabla_w)])
        self.weights = np.asarray([[w-(eta/len(mini_batch))*nw
                                  for w, nw in zip(warr, nwarr)]
                                 for warr, nwarr in zip(self.weights, nabla_w)])
        self.biases = np.asarray([b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)])

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        totalsamp = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, totalsamp, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \\partial C_x /
        \\partial a for the output activations."""
        return (output_activations-y)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

#### Miscellaneous functions
def sigmoid(z, numin):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z, numin):
    """Derivative of the sigmoid function."""
    return sigmoid(z,0)*(1-sigmoid(z,0))

def sigmoid_inverse(z, numin):
    # z = 0.998*z + 0.001
    assert(np.max(z) <= 1.0 and np.min(z) >= 0.0)
    z = 0.998*z + 0.001
    return np.log(np.true_divide(
        1.0, (np.true_divide(1.0, z) - 1)
    ))

def ReLU(z):
    result = np.array(z, numin)
    result[result < 0] = 0
    return np.divide(result, numin)

def ReLU_prime(z, numin):
    return np.divide((np.array(z) > 0).astype('int'),numin)

def GELU(z, numin):
    return np.multiply(np.multiply(1/(2*numin),z),
                       np.add(1, erf(np.divide(z,np.sqrt(2)))))

def GELU_prime(z, numin):
    cdf=np.multiply(1/(2*numin), np.add(1, erf(np.divide(z,np.sqrt(2)))))
    pdf=np.multiply(np.divide(z,numin), stdnormgauss(z))
    return np.add(cdf, pdf)
    
def stdnormgauss(z):
    return np.divide(np.exp(np.divide(np.power(z,2),-2)), np.sqrt(2*np.pi))

# Adds an arbitrary number of np arrays.
def npadd_many(firstarr, *nparrs):
    totalarr = firstarr
    for currarr in nparrs:
        totalarr = np.add(totalarr, currarr)
    return totalarr

def get_dzdn(layerweights, nactval, npos, zpos):
    total=0
    for n in range(len(layerweights)):
        total += layerweights[n][zpos][npos] * (n + 1) * np.power(nactval, n)
    return total
