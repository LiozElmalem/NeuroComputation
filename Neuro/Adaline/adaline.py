import matplotlib.pyplot as plt
import numpy as np
from random import seed
from csv import reader
from matplotlib.widgets import Button

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
 
# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def rand_bin_array(K, N):
    arr = np.zeros(N)
    arr[:K]  = 1
    np.random.shuffle(arr)
    return arr    

def printList(freeText , mylist):
    print (freeText , '[%s]' % ', '.join(map(str, mylist)))

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
    plt.plot(A,B)
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
        totalErrors = 0
        errors = []
        my_outputs = []        
        for x , target in zip(inputs , outputs):
            activate = activation(W , x , bias , treshold)
            my_outputs.append(activate)
            error = target - activate
            totalErrors += (error * error) # Mean square error
            update(W , learning_rate , error , x , bias)
        if(i % 100 == 0):
            printList('Iteration ' + str(i) + ' W : ' , W)  
            plot(my_outputs , outputs , 'Adaline simulation' , i , W , totalErrors)
        if(convergence(outputs , my_outputs)):
            plot(my_outputs , outputs , 'Adaline simulation' , i , W , totalErrors)
            print('The algorithm simulation arrived to convergence')
            exit(1)    
        errors.append(totalErrors)

def simulation():
    seed(1)
    # load and prepare data
    filename = 'C://Users//user//Desktop//Lioz//Neuro//Back_Propagation//seeds_dataset.csv'
    dataset = load_csv(filename)
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0])-1)
    # normalize input variables
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    learning_rate = 0.1
    treshold = 1
    bias = 1
    W = np.random.rand(2)
    inputs = []
    outputs = []
    for item in dataset:
        inputs.append(item[:2])
        outputs.append(item[len(item)-1])
    print('inputs ' , inputs)
    print('outputs , ' , outputs)
    for i in range(len(outputs)):
        if(outputs[i] == 2):
            outputs[i] = 0
    adaline(W , inputs , outputs , learning_rate , 1000 , treshold , bias)

if __name__ == '__main__':
    simulation()    
