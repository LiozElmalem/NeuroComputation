import numpy as np
import matplotlib.pyplot as plt
import math
from random import seed
from csv import reader

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

def treshold(x):
    if (x > 0):
        return 1
    else:
        return 0

def adaline(iterationAmount , INPUTS, OUTPUTS , WEIGHTS,errors,LEARNING_RATE):
    accuracy = 0
    for iter in range(iterationAmount):

        for input_item,desired in zip(INPUTS, OUTPUTS):
        
            # Feed this input forward and calculate the ADALINE output
            ADALINE_OUTPUT = 0
            for i in range(len(input_item)):
                ADALINE_OUTPUT += (input_item[i] * WEIGHTS[i])

            # Run ADALINE_OUTPUT through the step function
            ADALINE_OUTPUT = treshold(ADALINE_OUTPUT)

            # Calculate the ERROR generated
            ERROR = desired - ADALINE_OUTPUT
            
            # Store the ERROR
            errors.append(ERROR)
            
            # Update the weights based on the delta rule
            for i in range(len(WEIGHTS)):
                WEIGHTS[i] = WEIGHTS[i] + LEARNING_RATE * ERROR * input_item[i]
    
    print ("New Weights after training", WEIGHTS)
    for input_item,desired in zip(INPUTS, OUTPUTS):
    # Feed this input forward and calculate the ADALINE output
        ADALINE_OUTPUT = 0
        for i in range(len(input_item)):
            ADALINE_OUTPUT += (input_item[i] * WEIGHTS[i])

    # Run ADALINE_OUTPUT through the step function
        ADALINE_OUTPUT = treshold(ADALINE_OUTPUT)

        if(ADALINE_OUTPUT == desired):
            accuracy += 1 

        print ("Actual ", ADALINE_OUTPUT, "Desired ", desired)
    print('Accuary : ' , accuracy / len(OUTPUTS))

def show(errors):
    # Plot the errors to see how we did during training
    ax = plt.subplot(111)
    ax.plot(errors, c='#aaaaff', label='Training Errors')
    ax.set_xscale("log")
    plt.title("ADALINE Errors")
    plt.legend()
    plt.xlabel('Error')
    plt.ylabel('Value')
    plt.show()

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
    INPUTS = []
    OUTPUTS = []
    filedsAmount = 7
    for item in dataset:
        INPUTS.append(item[:filedsAmount])
        OUTPUTS.append(item[len(item)-1])
    for output in OUTPUTS:
        if(output == 2):
            output = 0  
    np.random.seed(1)
    WEIGHTS = 2 * np.random.random((filedsAmount,1)) - 1
    print ("Random Weights before training", WEIGHTS)
    LEARNING_RATE = 0.2
    errors = []
    adaline(1000 , INPUTS, OUTPUTS , WEIGHTS , errors,LEARNING_RATE)
    show(errors)

if __name__ == '__main__':
    simulation()    

