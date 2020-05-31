import string
import math
import random
from csv import reader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numpy as np

class Neural:
	def __init__(self):
		#
		# Lets take 2 input outputNodes, 3 hidden outputNodes and 1 output outputNode.
		# Hence, Number of outputNodes in input(ni)=2, hidden(hiddenNo)=3, output(outputNo)=1.
		#
		self.inputNo = 3
		self.hiddenNo = 3
		self.outputNo = 1

		#
		# outputNow we need outputNode weights. We'll make a two dimensional array that maps outputNode from one layer to the next.
		# i-th outputNode of one layer to j-th outputNode of the next.
		#
		self.wih = []
		for i in range(self.inputNo):
			self.wih.append([0.0] * self.hiddenNo)

		self.who = []
		for j in range(self.hiddenNo):
			self.who.append([0.0] * self.outputNo)

		#
		# outputNow that weight matrices are created, make the activation matrices.
		#
		self.ai, self.ah, self.ao = [],[],[]
		self.ai = [1.0] * self.inputNo
		self.ah = [1.0] * self.hiddenNo
		self.ao = [1.0] * self.outputNo

		#
		# To ensure outputNode weights are randomly assigned, with some bounds on values, we pass it through randomizeMatrix()
		#
		randomizeMatrix(self.wih,-0.2,0.2)
		randomizeMatrix(self.who,-2.0,2.0)

		#
		# To incorporate momentum factor, introduce aoutputNother array for the 'previous change'.
		#
		self.cih = []
		self.cho = []
		for i in range(self.inputNo):
			self.cih.append([0.0] * self.hiddenNo)
		for j in range(self.hiddenNo):
			self.cho.append([0.0] * self.outputNo)

	# backpropagate() takes as input, the patterns entered, the target values and the obtained values.
	# Based on these values, it adjusts the weights so as to balance out the error.
	# Also, outputNow we have M, N for momentum and learning factors respectively.
	def backpropagate(self, inputs, expected, output, N=0.5, M=0.1):
		# We introduce a new matrix called the deltas (error) for the two layers output and hidden layer respectively.
		output_deltas = [0.0] * self.outputNo
		for k in range(self.outputNo):
			# Error is equal to (Target value - Output value)
			error = expected[k] - output[k]
			output_deltas[k] = error * dsigmoid(self.ao[k])

		# Change weights of hidden to output layer accordingly.
		for j in range(self.hiddenNo):
			for k in range(self.outputNo):
				delta_weight = self.ah[j]  * output_deltas[k]
				self.who[j][k] += (M * self.cho[j][k]) + N*delta_weight
				self.cho[j][k] = delta_weight

		# outputNow for the hidden layer.
		hidden_deltas = [0.0] * self.hiddenNo
		for j in range(self.hiddenNo):
			# Error as given by formule is equal to the sum of (Weight from each outputNode in hidden layer times output delta of output outputNode)
			# Hence delta for hidden layer = sum (self.who[j][k]*output_deltas[k])
			error = 0.0
			for k in range(self.outputNo):
				error += self.who[j][k]  * output_deltas[k]
			# outputNow, change in outputNode weight is given by dsigmoid() of activation of each hidden outputNode times the error.
			hidden_deltas[j] = error * dsigmoid(self.ah[j])

		for i in range(self.inputNo):
			for j in range(self.hiddenNo):
				delta_weight = hidden_deltas[j] * self.ai[i]
				self.wih[i][j] += M*self.cih[i][j] + N*delta_weight
				self.cih[i][j] = delta_weight

	# Main testing function. Used after all the training and Backpropagation is completed.
	def test(self, patterns):
		accuary = 0
		for p in patterns:
			inputs = p[0]
			output = self.runNetwork(inputs)[0]
			output = (1 if (output > 0.5) else 0)
			target = p[1]
			accuary = (accuary + 1) if (output == target[0]) else accuary
			print ('For input:', inputs, ' Output -->', output, '\tTarget: ', target)
		print('Accuary : ' , accuary / len(patterns))

	# So, runNetwork was needed because, for every iteration over a pattern [] array, we need to feed the values.
	def runNetwork(self, feed):
		if(len(feed) != self.inputNo - 1):
			print ('Error in number of input values.')

		# First activate the ni-1 input outputNodes.
		for i in range(self.inputNo - 1):
			self.ai[i] = feed[i]

		#
		# Calculate the activations of each successive layer's outputNodes.
		#
		for j in range(self.hiddenNo):
			sum = 0.0
			for i in range(self.inputNo):
				sum += self.ai[i] * self.wih[i][j]
			# self.ah[j] will be the sigmoid of sum. # sigmoid(sum)
			self.ah[j] = sigmoid(sum)

		for k in range(self.outputNo):
			sum = 0.0
			for j in range(self.hiddenNo):
				sum += self.ah[j]*self.who[j][k]
			# self.ah[k] will be the sigmoid of sum. # sigmoid(sum)
			self.ao[k] = sigmoid(sum)
		return self.ao

	def train(self, pattern,iterations):
		for i in range(iterations):
			# Run the network for every set of input values, get the output values and Backpropagate them.
			for p in pattern:
				# Run the network for every tuple in p.
				inputs = p[0]
				out = self.runNetwork(inputs)
				expected = p[1]
				self.backpropagate(inputs,expected,out)

# End of class.

def on_click(event):
    if event.dblclick:
        exit(0)

def Exit(event):
    exit(1)

def plot(A):
	plt.title("BackPropagation")
	plt.plot(A)
	plt.xlabel('Iteration No.')
	plt.ylabel('Loss')
	plt.show()

def splitData(data , test , train):
    treshold = int(len(data) * 0.33)
    for i in (range(len(data))):
        test.append(data[i]) if(i < treshold) else train.append(data[i])
	
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

def getSeedDataSet():
    filename = './seeds_dataset.csv'
    dataset = load_csv(filename)
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0])-1)
    # normalize input variables
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    return dataset

def generate(test , train):	
	dataset = getSeedDataSet()
	pat = []
	for i in range(len(dataset)):
		pat.append([dataset[i][:2] , dataset[i][7:]])
	random.shuffle(pat)
	splitData(dataset,test,train)
	return pat


def randomizeMatrix ( matrix, a, b):
	for i in range ( len (matrix) ):
		for j in range ( len (matrix[i]) ):
			# For each of the weight matrix elements, assign a random weight uniformly between the two bounds.
			matrix[i][j] = random.uniform(a,b)


# outputNow for our function definition. Sigmoid.
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# Sigmoid function derivative.
def dsigmoid(y):
	return y * (1 - y)


def main():
	# take the input pattern as a map. Suppose we are working for AND gate.
	test = []
	train = []
	pattern = generate(test , train)
	newNeural = Neural()
	newNeural.train(pattern,500)
	newNeural.test(pattern)

if __name__ == "__main__":
	main()