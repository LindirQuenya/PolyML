import numpy as np

class Network(object):
    def __init__(self, sizes, polyorders, non_linearity = "sigmoid"):
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
        12th-order polynomials.The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        if np.isin(True, np.array(polyorders)<1):
            raise ValueError("Argument polyorders must have positive elements!")
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Zero-order weights. Not worth making a matrix out of.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # 1-n order weights. The zero-based indexing is deceptive here:
        # index zero corresponds to order 1, ind 1 -> ord 2, etc.
        self.weights = [[np.random.randn(y, x) for i in range(p)]
                        for x, y, p in zip(sizes[:-1], sizes[1:], polyorders)]
        if non_linearity == "sigmoid":
            self.non_linearity = sigmoid
            self.d_non_linearity = sigmoid_prime
        elif non_linearity == "ReLU":
            self.non_linearity = ReLU
            self.d_non_linearity = ReLU_prime
        elif non_linearity == "GELU":
            #TODO: implement this.
            raise ValueError("GELU not yet implemented.")
        else:
            raise ValueError("Invalid non_linearity")

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, warr in zip(self.biases, self.weights):
            # This looks complicated, kinda is. For each power p = 1..k, it multiplies
            # np.power(a, p) by the corresponding weight matrix. Because of the zero-based
            # indexing, this ends up being n = 0..(k-1), np.power(a, n+1), and warr[n].
            # In the above, k is the polynomial order for that connection. Also (conveniently)
            # the length of warr for a given connection set.
            a = self.non_linearity(npadd_many(b,*[np.dot(warr[n], np.power(a,n+1))
                                                  for n in range(len(warr))]))
        return a

    def get_activation_of_all_layers(self, input_a, n_layers = None, get_zs = False):
        """Gets the activations of the neurons and, if get_zs is True, their pre-non_linearity values."""
        if n_layers is None:
            n_layers = self.num_layers
        activations = [input_a.reshape((input_a.size, 1))]
        if get_zs:
            zs=[]
        for b, warr in list(zip(self.biases, self.weights))[:n_layers]:
            last_a = activations[-1]
            # See the comment in Network.feedforward() for an explaination of this.
            z = npadd_many(b,*[np.dot(warr[n], np.power(last_a,n+1)) for n in range(len(warr))])
            if get_zs:
                zs.append(z)
            new_a = self.non_linearity(z)
            new_a = new_a.reshape((new_a.size, 1))
            activations.append(new_a)
        if get_zs:
            return activations, zs
        else:
            return activations

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \\partial C_x /
        \\partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def sigmoid_inverse(z):
    # z = 0.998*z + 0.001
    assert(np.max(z) <= 1.0 and np.min(z) >= 0.0)
    z = 0.998*z + 0.001
    return np.log(np.true_divide(
        1.0, (np.true_divide(1.0, z) - 1)
    ))

def ReLU(z):
    result = np.array(z)
    result[result < 0] = 0
    return result

def ReLU_prime(z):
    return (np.array(z) > 0).astype('int')

# Adds an arbitrary number of np arrays.
def npadd_many(firstarr, *nparrs):
    totalarr = firstarr
    for currarr in nparrs:
        totalarr = np.add(totalarr, currarr)
    return totalarr

net1 = Network([2,2,1], [3,3])
x = np.array([0.5, 0.2])
y = np.array([0.3])
