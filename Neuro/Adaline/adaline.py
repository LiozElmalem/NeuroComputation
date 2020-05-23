import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

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

def update(W , learning_rate , error , input_ , bias):    
    for i in range(len(W)):
        W[i] = W[i] + (learning_rate * error * input_[i])
    bias = bias + (learning_rate * error)

def on_click(event):
    if event.dblclick:
      exit(0)

def Exit(event):
    exit(1)

def plot(A , B , title , iterationNo , W , errors):
    plt.plot(A, B)
    plt.ylabel('Targets')
    plt.xlabel('Outputs')
    plt.title(title + ' iteration No. ' + str(iterationNo) + ' total mean square error ' + str(errors))
    plt.connect('button_press_event', on_click)
    button_position = plt.axes([0.9, 0.0, 0.1, 0.075])
    button = Button(button_position, 'Exit', color='lightblue', hovercolor='lightgreen')
    button.on_clicked(Exit)
    plt.show()  

def test(outputs , W , inputs , treshold,bias):
    positives = 0
    for i , j in zip(inputs , outputs):
        y = activation(W , i , bias , treshold)
        if(y == j):
            positives += 1
    accuary = positives / len(inputs)        
    return accuary

def convergence(targets , outputs):
    flag = True
    for i in range(len(targets)):
        if(targets[i] != outputs[i]):
            flag = False
    return flag

def adaline(W , inputs , outputs , learning_rate , iterationAmount , treshold , bias):
    for i in range(iterationAmount):
        errors = 0
        my_outputs = []        
        for x , target in zip(inputs , outputs):
            activate = activation(W , x , bias , treshold)
            my_outputs.append(activate)
            error = target - activate
            errors += (error * error) # Mean square error
            update(W , learning_rate , error , x , bias)
        printList('Iteration ' + str(i) + ' W : ' , W)  
        if(i % (iterationAmount / 10) == 0):
            plot(my_outputs , outputs , 'Adaline simulation' , i , W , errors)
        if(convergence(outputs , my_outputs)):
            plot(my_outputs , outputs , 'Adaline simulation' , i , W , errors)
            print('The algorithm simulation arrived to convergence')
            exit(1)    
    return errors        

def simulation():
    # data_ = init(5 , 50)

    # W = data_[2]
    # outputs = data_[1]
    # inputs = data_[0]
    learning_rate = 0.1
    treshold = 0.5
    bias = 0.1
    W = [0.5 , 0.5]
    inputs = [
        [0 , 0],
        [0 , 1],
        [1 , 0],
        [1 , 1]
    ]
    outputs = [0 , 0 , 0 , 1]
    adaline(W , inputs , outputs , learning_rate , 100 , treshold , bias)

if __name__ == '__main__':
    simulation()    
