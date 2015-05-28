import numpy as np
from numpy import dot, outer
import random
import sys
import copy

def randmatrix(m, n, lower=-0.5, upper=0.5):
    """Creates an m x n matrix of random values in [lower, upper]"""
    return np.array([random.uniform(lower, upper) for i in range(m*n)]).reshape(m, n)

class ShallowNeuralNetwork:
    def __init__(self, input_dim=0, hidden_dim=0, output_dim=0, afunc=np.tanh, d_afunc=(lambda z : 1.0 - z**2)):
        self.afunc = afunc
        self.d_afunc = d_afunc
        self.input = np.ones(input_dim+1)   # +1 for the bias
        self.hidden = np.ones(hidden_dim+1) # +1 for the bias
        self.output = np.ones(output_dim)
        self.iweights = randmatrix(input_dim+1, hidden_dim)
        self.oweights = randmatrix(hidden_dim+1, output_dim)
        self.oerr = np.zeros(output_dim)
        self.ierr = np.zeros(input_dim+1)

    def forward_propagation(self, ex):
        self.input[ : -1] = ex # ignore the bias
        self.hidden[ : -1] = self.afunc(dot(self.input, self.iweights)) # ignore the bias
        self.output = self.afunc(dot(self.hidden, self.oweights))
        return copy.deepcopy(self.output)

    def backward_propagation(self, labels, alpha=0.5):
        labels = np.array(labels)
        self.oerr = (labels-self.output) * self.d_afunc(self.output)
        herr = dot(self.oerr, self.oweights.T) * self.d_afunc(self.hidden)
        self.oweights += alpha * outer(self.hidden, self.oerr)
        self.iweights += alpha * outer(self.input, herr[:-1]) # ignore the bias
        return np.sum(0.5 * (labels-self.output)**2)

    def train(self, training_data, training_labels, maxiter=5000, alpha=0.05, epsilon=1.5e-8, display_progress=False):
        iteration = 0
        error = sys.float_info.max
        while error > epsilon and iteration < maxiter:
            error = 0.0
            random.shuffle(training_data)
            for i in range(len(training_data)):
                ex = training_data[i]
                labels = training_labels[i]
                self.forward_propagation(ex)
                error += self.backward_propagation(labels, alpha=alpha)
            if display_progress:
                print 'completed iteration %s; error is %s' % (iteration, error)
            iteration += 1

    def predict(self, ex):
        self.forward_propagation(ex)
        return copy.deepcopy(self.output)

    def hidden_representation(self, ex):
        self.forward_propagation(ex)
        return self.hidden

    def score(self, data, labels):
        score = 0
        for i in range(len(data)):
            prediction = self.predict(data[i])
            predictionLabel = np.argmax(prediction)
            # print(labels[i], prediction)
            if np.argmax(labels[i]) == predictionLabel:
                score += 1
        return (score *1.0)/len(data)
