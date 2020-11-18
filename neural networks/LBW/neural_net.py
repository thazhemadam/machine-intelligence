'''
	IMPLEMENTATION DETAILS & SPECIFICATIONS
 
	Dataset Cleaning & Preprocessing:
		The dataset has missing values with the distribution as given below:
		Community		 0
		Age		 		 7
		Weight			11
		Delivery phase	 4
		HB				19
		IFA		 		 0
		BP				15
		Education	 	 3
		Residence	 	 2
		Result		 	 0
		dtype: int64

		'Age' has 14 unique values, and it shows slight left skewness. So the missing values for 'Age' is filled with the median.
		'Delivery Phase' shows skewness toom and has the missing values are filled with the median.
  		'HB' and 'Weight' features show right skewness and are therefore filled with the median.
		'Education' has only one unique value 5 and hence the missing values are filled with mode. (For this reason, ‘Education’ obviously isn’t a significant hyperparameter.)
		'BP' is symmetric and therefore the missing values were directly filled with the mean value.
		'Residence' has only 3 unique values and only 2 NaN's and therefore it is filled with 0.
		


	NN ARCHITECTURE: 

	Total number of layers used: 5
	Number of Hidden Layers: 4
	Output layer: 1
	The number of perceptrons at the input: 9 (9 input features on which training takes place)
	The number of perceptrons at the hidden layer 1: 7
	The number of perceptrons at the hidden layer 2: 5
	The number of perceptrons at the hidden layer 3: 4
	The number of perceptrons at the hidden layer 4: 7
	The number of perceptrons at the output layer: 1 (Binary output (0 or 1))
	The learning rate: 0.00095
	The number of epochs: 1000
	Weight Initialisation: Random Normal Distribution
	Optimisation: Batch Gradient Descent

	Dimensions of the Weight Matrices: 
		W1: 9 x 7 (input and hidden layer 1)
		W2: 7 x 5 (hidden layer 1 and hidden layer 2)
		W3: 5 x 4 (hidden layer 2 and hidden layer 3)
		W4: 4 x 7 (hidden layer 3 and hidden layer 4)
		W5: 7 x 1 (hidden layer 4 and the output layer(layer 5))

	Dimensions of the Bias Matrices:
		B1: 7 x 1 (bias for each of the 7 perceptrons in the hidden layer 1)
		B2: 5 x 1 (bias for each of the 5 perceptrons in the hidden layer 2)
		B3: 4 x 1 (bias for each of the 4 perceptrons in the hidden layer 3)
		B4: 7 x 1 (bias for each of the 7 perceptrons in the hidden layer 3)
		B5: 1 x 1 (bias for the perceptron in the output layer)

	Activation Function:
		Hidden Layer 1: ReLU
		Hidden Layer 2: ReLU
		Hidden Layer 3: ReLU
		Hidden Layer 4: ReLU
		Output Layer: sigmoid_activation (To get the output to a probabililty(sum is 1) that classifies the input to either of the 2 classes)

	Loss Function:
		Binary Cross Entropy a.k.a. negative log likelihood function
		BC(y,yhat) = −(ylog(yhat) + (1-y)log(1-yhat)) where y and yhat are probability distributions
		-> y is the actual output classification
		-> yhat is the predicted output classification

	Plot of the Loss Curve:
		Plot the loss curve by keeping track of the loss at each epoch

	Sigmoid Activation Function and Binary Cross entropy Loss Functions:
		Take care of overflow and underflow conditions and add a small lambda for yhat=0 or yhat=1 to handle log(0) errors

	Forward Propagation:
		for each layer i:
			y = (input*weights + bias) for the layer
			if layer = last layer:
				yhat = sigmoid(y)
				break
			else:
				input  = ReLU(y)
		calculate BinaryCrossEntropy loss 

	Back Propagation:
		BC(y,yhat)= −(ylog(yhat) + (1-y)log(1-yhat))
		derivative of BC wrt to W(FinalHiddenLayer) = derivative of BC wrt to yhat * derivative of yhat wrt net * derivative of net wrt W(FinalHiddenLayer) 
		derivative of BC wrt to yhat = - (y/yhat - (1-y)/(1-yhat))
		derivative of yhat wrt net = derivative of ( 1/1+exp(-net)) wrt net = yhat*(1-yhat)
		derivative of net wrt W(FinalHiddenLayer) = A = (input to the last layer)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean(dataset):
	cleaned_dataset = 'CleanedLBW_Dataset.csv'
	data = pd.read_csv(dataset)
	data['Age'] = data['Age'].fillna(data['Age'].median())
	data['Delivery phase'] = data['Delivery phase'].fillna(data['Delivery phase'].median())
	data['Education'] = data['Education'].fillna(data['Education'].mode()[0])
	data['Residence'] = data['Residence'].fillna(0)
	data['BP'] = data['BP'].fillna(data['BP'].dropna().mean())
	data['HB'] = data['HB'].fillna(data['HB'].dropna().median())
	data['Weight'] = data['Weight'].fillna(data['Weight'].dropna().median())
	data.to_csv(cleaned_dataset, index=False)
	return cleaned_dataset

class NN:

	''' X and Y are dataframes '''
	def __init__(self, layers=[9,7,5,8,1], lr=0.001, epochs=1000):
		self.input = None
		self.epochs = epochs
		self.label = None
		self.layers = layers
		self.lossBC = []
		self.lr = lr
		self.parameters = dict()

	# Initialize weights from a random normal distribution
	def init_weights(self,layer_count):
		'''
			parameters['Wx'] for x in [1,2,..,layers] defines the weight matrix for layer x
			parameters['bx'] for x in [1,2,..,layers] defines the bias matrix for layer x
		'''

		# Seed the random number generator
		np.random.seed(1)
		for layer in range(0,layer_count-1):
			self.parameters['W'+str(layer+1)] = np.random.randn(self.layers[layer], self.layers[layer+1])
			self.parameters['b'+str(layer+1)] = np.random.randn(self.layers[layer+1],)

	# ReLU for intermediate layers: Positive thresholding
	def ReLU(self, Y):
		return np.maximum(0, Y)

	# Derivative of the ReLU function
	def ReLUDerivative(self,x):
		x[x <= 0] = 0
		x[x > 0] = 1
		return x

	# Sigmoid to convert it to a range between 0 and 1
	def sigmoid_activation(self, Y):
		# Prevent overflow and underflow
		res = np.zeros(Y.shape,dtype='float')
		for i in range(0, len(Y)):
			if -Y[i] > np.log(np.finfo(float).max):
				res[i]= 0.0 
			elif Y[i] > np.log(np.finfo(float).max):
				res[i] = 0.0
			else:
				res[i]=1.0 / (1.0 + np.exp(-Y[i])) 
		return res

	# Binary Cross Entropy Loss Function
	def bce_loss(self,y,yhat):
		'''
			y is the actual output classification
			yhat is the predicted output classification
		'''
		number = len(y)

		# Add a small lambda value to handle of log(0) 
		for i in range(len(yhat)): 
			if yhat[i] == 0:
				yhat[i] = 1e-7
			elif yhat[i] == 1:
				yhat[i] -= 1e-7

		'''Binary CrossEntropy Function'''
		loss = -1/number * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((1 - y), np.log(1 - yhat)))) 
		return loss

	# Forward Propagation
	def forward_propagation(self, layer_count):
		'''
		   layer_count: number of layers in the NN
		   self.input: features used for training

		   L stores the linear combination (input*W + b)
		   A stores the result of activation applied to L. This is given as the input to the next layer
		'''
		L1 = self.input.dot(self.parameters['W1']) + self.parameters['b1']
		A1 = self.ReLU(L1)
		self.parameters['L1'] = L1
		self.parameters['A1'] = A1
		for i in range(2, layer_count-1):
			L1 = A1.dot(self.parameters['W' + str(i)]) + self.parameters['b' + str(i)]
			A1 = self.ReLU(L1)
			self.parameters['L' + str(i)] = L1
			self.parameters['A' + str(i)] = A1
		L2 = A1.dot(self.parameters['W' + str(layer_count-1)]) + self.parameters['b' + str(layer_count-1)]
		yhat = self.sigmoid_activation(L2)
		loss = self.bce_loss(self.label, yhat)
		self.parameters['L' + str(layer_count-1)] = L2

		return yhat, loss

	# Backward Propagation that simultaneously updates weights and biases
	def backpropagation(self, yhat, layer_count):

		'''
			yhat: the predicted output classification 
			diff_yhat: derviative of loss/error wrt yhat
			diff_sigmoid: derviative of yhat wrt to non-activated previous linear combination (sigmoid derviative)
			diff_z: diff_yhat * diff_sigmoid
		'''
		diff_yhat = -(np.divide(self.label,yhat) - np.divide((1 - self.label),(1-yhat)))

		# Sigmoid derivative for the last layer
		diff_sigmoid = yhat * (1-yhat) 
		diff_z = diff_yhat * diff_sigmoid

		for i in range(layer_count-1, 1, -1):
			diffA1 = diff_z.dot(self.parameters['W' + str(i)].T)
			diffw2 = self.parameters['A' + str(i-1)].T.dot(diff_z)
			diffb2 = np.sum(diff_z, axis=0)
			
			# Update weights and biases for intermediate layers
			self.parameters['W' + str(i)] = self.parameters['W' + str(i)] - self.lr * diffw2
			self.parameters['b' + str(i)] = self.parameters['b' + str(i)] - self.lr * diffb2

			diffReLU = self.ReLUDerivative(self.parameters['A'+str(i-1)]) 
			diff_z = diffA1 * diffReLU


		diff_z1 = diffA1 * self.ReLUDerivative(self.parameters['L1'])
		diffw1 = self.input.T.dot(diff_z1)
		diffb1 = np.sum(diff_z1, axis=0)

		# Updation weights and bias for layer 1
		self.parameters['W1'] = self.parameters['W1'] - self.lr * diffw1
		self.parameters['b1'] = self.parameters['b1'] - self.lr * diffb1

	# Performance Metric: Accuracy
	def accuracy(self, y, yhat):
		accuracy = (sum(y == yhat) / len(y) * 100)
		return accuracy


	def fit(self, X, y):
		'''
			X: input features for training
			y: label for classification output
		'''

		self.input = X
		self.label = y

		# Initialize weights and bias
		self.init_weights(len(self.layers)) 


		for i in range(self.epochs):
			yhat, loss = self.forward_propagation(len(self.layers))
			self.backpropagation(yhat, len(self.layers))
			self.lossBC.append(loss)

	# Predicting using Test Data
	def predict(self,X):
		for i in range(1,len(self.layers)):
			L1 = X.dot(self.parameters['W' + str(i)]) + self.parameters['b' + str(i)]
			A1 = self.ReLU(L1)
			X=A1
		predicted = self.sigmoid_activation(L1)
		yhat = np.round(predicted)  
		return yhat

	# Print the Confusion Matrix
	def CM(self, y_test, y_test_obs):
		'''
			y_test is list of y values in the test dataset
			y_test_obs is list of y values predicted by the model

		'''
		for i in range(len(y_test_obs)):
			if(y_test_obs[i] > 0.6):
				y_test_obs[i] = 1
			else:
				y_test_obs[i] = 0

		cm = [[0,0],[0,0]]
		fp = 0
		fn = 0
		tp = 0
		tn = 0

		for i in range(len(y_test)):
			if(y_test[i] == 1 and y_test_obs[i] == 1):
				tp = tp + 1
			if(y_test[i]== 0 and y_test_obs[i] == 0):
				tn = tn + 1
			if(y_test[i] == 1 and y_test_obs[i] == 0):
				fp = fp + 1
			if(y_test[i] == 0 and y_test_obs[i] == 1):
				fn = fn + 1
		cm[0][0] = tn
		cm[0][1] = fp
		cm[1][0] = fn
		cm[1][1] = tp

		p = tp/(tp + fp)
		r = tp/(tp + fn)
		f1 = (2*p*r)/(p + r)

		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")

	# Plot the loss function
	def loss_plot(self):
		plt.title("Loss Plot")
		plt.plot(self.lossBC)
		plt.xlabel("Number of Epochs")
		plt.ylabel("Loss")
		plt.show()

def neural_network():

	# Data cleaning
	lbw_dataset = "LBW_Dataset.csv"
	cleaned_dataset = clean(lbw_dataset)
	data = pd.read_csv(cleaned_dataset)

	# Data extraction
	X = data.drop(columns=['Result'])
	y = data['Result'].values.reshape(X.shape[0], 1)

	# Stratified distribution splitting of the dataset
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42, stratify = y, shuffle = True)

	# Preprocessing the data: Scaling and Normalising the data
	sc = StandardScaler()
	sc.fit(X_train)
	X_train = sc.transform(X_train)
	X_test = sc.transform(X_test)

	# Building the NN model
	neural_network = NN(layers=[9, 7, 5, 4, 7, 1], lr = 0.00095, epochs = 1000)

	# Training the model to fit the data
	neural_network.fit(X_train, y_train) 

	train_pred = neural_network.predict(X_train)
	test_pred = neural_network.predict(X_test)

	print("Train accuracy : {}".format(neural_network.accuracy(y_train, train_pred)))
	print("Test accuracy : {}".format(neural_network.accuracy(y_test, test_pred)))
	neural_network.CM(y_test, test_pred)

	# Plot the loss function
	neural_network.loss_plot()

neural_network()
