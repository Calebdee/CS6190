from scipy.special import expit as logistic
import pandas as pd 
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np

tolerance = 1e-5
max_iterations = 100

train = pd.read_csv("/Users/calebjohnson/Desktop/Code/pml/hw2/CS6190/data/bank-note/train.csv")
test  = pd.read_csv("/Users/calebjohnson/Desktop/Code/pml/hw2/CS6190/data/bank-note/test.csv")
trainY = train.iloc[:,-1]
testY = test.iloc[:,-1]
trainX = train.iloc[:,0:-1]
testX = test.iloc[:,0:-1]

weight = np.zeros((trainX.shape[1], 1))

for i in range(max_iterations):
	dot_product = np.matmul(trainX,weight)
	pred = output = logistic(dot_product)
	gradient = np.matmul(trainX.T, pred - trainY)

	R = np.eye(pred.shape[0])
	for i in range(pred.shape[0]):
			R[i,i] = pred[i,0] * (1- pred[i,0])
	hessian = np.matmul(np.matmul(trainX.T, R), trainX) + np.eye(trainX.shape[1])/reg

	weight_new = weight - np.matmul(np.linalg.inv(hessian), gradient)
	
	diff = np.linalg.norm(weight_new- weight)

	acc, _ = predict_and_test(weight_new, testX, testY)

	weight = weight_new

	if diff < tolerance:
		print("Converged")
		break