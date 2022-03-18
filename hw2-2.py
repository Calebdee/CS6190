print("Using Newton Raphson Solver\n")
for i in range(max_iter):
	# Get training data prediction
	pred, dot_product = get_output(weight, train_data, regression= regression)
	# Get gradient and hessian
	gradient = get_gradient(phi= train_data, pred= pred, t= train_label[:, np.newaxis], dot_product= dot_product, weight= weight, reg= reg, regression= regression)
	hessian  = get_hessian (phi= train_data, pred= pred, t= train_label[:, np.newaxis], dot_product= dot_product, reg= reg, regression= regression)
	# Update weights
	weight_new = weight - np.matmul(np.linalg.inv(hessian), gradient)
	
	# Difference between weights
	diff = np.linalg.norm(weight_new- weight)

	# Get accuracy
	acc, _ = predict_and_test(weight_new, test_data, test_label, regression= regression)

	weight = weight_new

	print("Iteration= {:3d} Diff_in_weight= {:.5f} Test_Acc= {:.2f}%".format(i, diff, acc))
	if diff < tolerance:
		#print(weight)
		print("Training converged. Done.")
		break