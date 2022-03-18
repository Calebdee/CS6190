from scipy.special import expit as logistic
import pandas as pd 
from scipy.stats import norm
from scipy.optimize import minimize

tolerance = 1e-5
max_iterations = 100

train = pd.read_csv("/Users/calebjohnson/Desktop/Code/pml/hw2/CS6190/data/bank-note/train.csv")
test  = pd.read_csv("/Users/calebjohnson/Desktop/Code/pml/hw2/CS6190/data/bank-note/test.csv")

def get_output(weight, data, regression= "logistic"):
	dot_product = np.matmul(data,weight)
	if regression == "logistic":
		output = logistic(dot_product)
	elif regression == "probit":
		output = norm.cdf(dot_product)
	elif regression == "multiclass":
		output = softmax(dot_product, axis=1)

	return output, dot_product

def get_accuracy(pred, test_label, regression= "logistic"):
	if regression == "multiclass":
		pred_max = np.argmax(pred, axis=1)
		gt_max   = np.argmax(test_label, axis=1)
		acc = np.sum(pred_max == gt_max)*100.0/pred.shape[0]
	elif regression == "logistic" or regression == "probit":
		if pred.ndim == 2:
			pred = pred[:,0]
		pred[pred > 0.5] = 1.0
		pred[pred < 1.0] = 0.0
		acc = np.sum(pred == test_label)*100.0/pred.shape[0]

	return acc

def predict_and_test(weight, test_data, test_label, regression= "logistic"):
	# Predict on test data and get accuracy
	pred_test, _ = get_output(weight, test_data, regression= regression)
	acc          = get_accuracy(pred_test, test_label, regression= regression)
	return acc, pred_test

def get_gradient(phi, pred, t, dot_product, weight, reg= 1, regression= "logistic"):
	if regression == "logistic":
		gradient = np.matmul(phi.T, pred - t)
	elif regression == "probit":
		R = np.eye(pred.shape[0])
		for i in range(pred.shape[0]):
			y_n  = pred[i,0]
			dotp = dot_product[i, 0]
			pdf  = norm.pdf(dotp)
			R[i,i] = pdf/(y_n*(1-y_n) + TOLERANCE)
		gradient = np.matmul(np.matmul(phi.T, R), pred-t)
	elif regression == "multiclass":
		gradient = np.matmul(phi.T, pred - t)

	# Add regularization
	gradient += weight/ reg
	return gradient

def get_hessian(phi, pred, t, dot_product, reg= 1, regression= "logistic"):
	R = np.eye(pred.shape[0])
	if regression == "logistic":
		for i in range(pred.shape[0]):
			R[i,i] = pred[i,0] * (1- pred[i,0])
	elif regression == "probit":
		for i in range(pred.shape[0]):
			y_n  = pred[i,0]
			t_n  = t[i,0] 
			dotp = dot_product[i, 0]
			pdf  = norm.pdf(dotp)

			term1 = 1/ (y_n * (1- y_n) + TOLERANCE)
			term2 = (y_n - t_n)/(y_n**2 * (1- y_n) + TOLERANCE)
			term3 = (y_n - t_n)/((1- y_n)**2 * y_n + TOLERANCE)
			term4 = (y_n - t_n)* dotp/(y_n * (1- y_n) * pdf + TOLERANCE)

			R[i,i] = (term1 - term2 + term3 - term4)*(pdf**2)

	# Add regularization			
	hessian = np.matmul(np.matmul(phi.T, R), phi) + np.eye(phi.shape[1])/reg
	return hessian	

print("Using Newton Raphson Solver\n")
for i in range(max_iterations):
	pred, dot_product = get_output(weight, train_data, regression= regression)
	gradient = get_gradient(phi= train_data, pred= pred, t= train_label[:, np.newaxis], dot_product= dot_product, weight= weight, reg= reg, regression= regression)
	hessian  = get_hessian (phi= train_data, pred= pred, t= train_label[:, np.newaxis], dot_product= dot_product, reg= reg, regression= regression)
	weight_new = weight - np.matmul(np.linalg.inv(hessian), gradient)
	
	diff = np.linalg.norm(weight_new- weight)

	acc, _ = predict_and_test(weight_new, test_data, test_label, regression= regression)

	weight = weight_new

	print("Iteration= {:3d} Diff_in_weight= {:.5f} Test_Acc= {:.2f}%".format(i, diff, acc))
	if diff < tolerance:
		#print(weight)
		print("Training converged. Done.")
		break