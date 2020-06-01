import pandas as pd
import math
import random
import numpy as np
import timeit
import matplotlib.pyplot as plt

class Adaline:
    def __init__(self , weights , bias = 0.1 , alpha = 0.1):
        self.weights = weights
        self.bias = bias
        self.alpha = alpha

    def treshold(self,x):
        return 1 if x >= 0 else -1

    def activate(self , item):
        return np.dot(item,self.weights) + self.bias

    def update(self , error , data):
        self.weights += self.alpha * data.T.dot(error)
        self.bias += (self.alpha * error)

    def train(self , epochs , INPUTS , OUTPUTS):
        i = 0
        acc = []
        while(i < epochs):
            errors = 0
            for item , target in zip(INPUTS , OUTPUTS):
                error = (target - self.activate(item))
                self.update(error , item)
                errors += (error**2)
            i += 1    
            acc.append(errors / 2.0)
        plt.plot(acc)
        plt.title('Adaline simulation on WPBC')
        plt.xlabel('Iterations')
        plt.ylabel('Mean squared error')
        plt.show()

    def test(self , INPUTS , OUTPUTS):
        accuracy = 0
        for i in range(len(OUTPUTS)):
            if(self.treshold(self.activate(INPUTS[i])) == OUTPUTS[i]):
                accuracy += 1 
        return accuracy / (float)(len(OUTPUTS)) 

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def main():

    df = pd.read_csv('./wpbc.data', header=None)
    start = 7
    fildes_num = 8 

    test_percent = 0.50
    test_size = int(test_percent * (197))
    train_size = 197 - test_size

    train_outputs = df.iloc[:train_size, 1].values
    train_outputs = np.where(train_outputs == 'N', -1, 1)
    train_inputs = df.iloc[:train_size, start:(start + fildes_num)].values

    test_outputs = df.iloc[train_size:(train_size + test_size), 1].values
    test_outputs = np.where(test_outputs == 'N', -1, 1)
    test_inputs = df.iloc[train_size:(train_size + test_size), start:(start + fildes_num)].values

    ada = Adaline(weights = [0.1] * fildes_num , bias = 0.1 , alpha = 0.1)
   
    start = timeit.default_timer()

    ada.train(100 ,train_inputs, train_outputs)

    stop = timeit.default_timer()

    print('Time ', stop - start)
    
    print('Accuracy ' , ada.test(test_inputs , test_outputs) * 100 , '%')

if __name__ == '__main__':
    main()