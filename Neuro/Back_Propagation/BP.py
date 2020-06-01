import math
import random
from csv import reader
import numpy as np
import timeit
import matplotlib.pyplot as plt

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a) * random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.1, 0.1)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-1.0, 1.0)

        # last change in weights for momentum
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        acc = 0
        for p in patterns:
            if((1 if self.update(p[0])[0] > 0.5 else -1) == p[1][0]):
                acc += 1
        print('Accuary ' , acc / float(len(patterns)) * 100 , '%')

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.1, M=0.1):
        # N: learning rate
        # M: momentum factor
        start = timeit.default_timer()
        errors = []
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            errors.append(error)
            if i % 100 == 0:
                print('error %-.5f' % error)
        stop = timeit.default_timer()
        print('Train time ', stop - start)
        plt.title('Back propagation')
        plt.xlabel('Epochs')
        plt.ylabel('Mean square error')
        plt.plot(range(iterations),errors)
        plt.show()      

def load_dataset(dataset_path, n_train_data , vars):
    dataset = []
    with open(dataset_path, 'r') as file:
        csv_reader = reader(file, delimiter=',')
        for row in csv_reader:
            item = [None] * vars
            item[0:vars-1] = list(map(float, row[9:9+vars-1]))
            item[vars-1] = (1 if row[1] == 'N' else -1)
            dataset.append([item[0:vars-1],[item[vars-1]]])
    train_data = dataset[0:n_train_data]
    val_data = dataset[n_train_data:]

    return train_data, val_data

def demo():

    train_size = 0.50
    fields = 3
    file_path = './wpbc.data'
    train_data, val_data = load_dataset(file_path, int(train_size * 198),fields)
    # create a network with two input, two hidden, and one output nodes
    n = NN(2, 2, 1)
    # train it with some patterns
    n.train(train_data , iterations=10000 , N=0.1 , M=0.1)
    # test it
    n.test(val_data)




if __name__ == '__main__':
    demo()