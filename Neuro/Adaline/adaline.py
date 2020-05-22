import matplotlib.pyplot as plt
import numpy as np

learning_rate = 0.2
treshold = 1
bias = 0.5

# or
def rand_bin_array(K, N):
    arr = np.zeros(N)
    arr[:K]  = 1
    np.random.shuffle(arr)
    return arr    

def printList(freeText , mylist):
    print (freeText , '[%s]' % ', '.join(map(str, mylist)))

def init(dimension , amount):
    data = np.random.rand(amount,dimension)
    targets = rand_bin_array(amount // 2 , amount)
    W = np.random.rand(dimension)
    return [data , targets , W]

def activation(W , input_ , bias , treshold):
    y = 0
    for x , weight in zip(input_ , W):
        y += x * weight 
    y += bias    
    if(y >= treshold):
        return 1
    else:
        return 0    

def update(W , learning_rate , error , input_):    
    for i in range(len(W)):
        W[i] = W[i] + (learning_rate * error * input_[i])

def plot(A , B):
    plt.plot(A, B)
    plt.ylabel('Targets')
    plt.xlabel('Outputs')
    plt.title('Adaline learning algorithm')
    plt.show()

def test(outputs , W , inputs , treshold):
    accuary = 0
    for i , j in zip(inputs , outputs):
        y = activation(W , i , bias , treshold)
        if(y == j):
            accuary += 1
    return accuary / len(inputs)

def convergence(targets , outputs):
    flag = True
    for i in range(len(targets)):
        if(targets[i] != outputs[i]):
            flag = False
    return flag

def adaline(W , inputs , outputs , learning_rate , iterationAmount , treshold):
    for i in range(iterationAmount):
        my_outputs = []        
        for x , target in zip(inputs , outputs):
            activate = activation(W , x , bias , treshold)
            my_outputs.append(activate)
            error = target - activate
            update(W , learning_rate , error , x)  
        if(i % (iterationAmount / 10) == 0):
            printList('targets : ' , outputs)
            printList('outputs : ' , my_outputs)
            plot(my_outputs , outputs)
        if(convergence(outputs , my_outputs)):
            exit(0)   

def simulation():
    data_ = init(5 , 50)

    W = data_[2]
    outputs = data_[1]
    inputs = data_[0]
    adaline(W , inputs , outputs , learning_rate , 10000 , treshold)

if __name__ == '__main__':
    simulation()    
