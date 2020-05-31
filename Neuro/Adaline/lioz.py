import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt


class Adaline:
    def __init__(self , weights , bias = 0.1 , alpha = 0.1):
        self.weights = weights
        self.bias = bias
        self.alpha = alpha

    def treshold(self,x):
        return 1 if x > 0 else -1

    def activate(self , item):
        temp = 0.0
        for i in range(len(item)):
            temp += self.weights[i] * item[i]
        return self.treshold(temp + self.bias)

    def update(self , error , data):
        for i in range(len(data)):
            self.weights[i] = self.weights[i] + (self.alpha * error * data[i])
        self.bias = self.bias + (self.alpha * error)

    def train(self , epochs , INPUTS , OUTPUTS):
        i = 0
        acc = []
        while(i < epochs):
            errors = 0
            for item , target in zip(INPUTS , OUTPUTS):
                error = target - self.activate(item)
                if(error != 0):
                    self.update(error , item)
                errors += (error**2)
            i += 1    
            acc.append(errors)
        plt.plot(acc)
        plt.title('Adaline simulation on WPBC')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.show()

    def test(self , INPUTS , OUTPUTS):
        accuary = 0
        for i in range(len(OUTPUTS)):
            if(self.activate(INPUTS[i]) == OUTPUTS[i]):
                accuary += 1 
        return accuary / (float)(len(OUTPUTS)) 

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def main():

    df = pd.read_csv('./wpbc.data', header=None)

    train_outputs = df.iloc[:120, 1].values
    train_outputs = np.where(train_outputs == 'N', -1, 1)
    train_inputs = df.iloc[:120, 3:34].values

    test_outputs = df.iloc[120:197, 1].values
    test_outputs = np.where(test_outputs == 'N', -1, 1)
    test_inputs = df.iloc[120:197, 3:34].values

    ada = Adaline(weights = [0.5] * 31 , bias = 0.1 , alpha = 0.1)

    ada.train(100 ,train_inputs, train_outputs)

    print('Accuary ' , ada.test(test_inputs , test_outputs) * 100 , '%')

main()