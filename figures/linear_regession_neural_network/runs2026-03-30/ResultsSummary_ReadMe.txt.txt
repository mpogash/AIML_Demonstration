	----------------------------
	2026-03-30
	Mike Pogash
	----------------------------

	A parametric study was performed by varying the number of epochs
	and the learning rate of the tensor flow linear regression model. 
		Number of Epochs: 10, 50, 100
		Learning Rate: 0.1, 0.01
		Caveats: Omitted sample at Number of Epochs = 100 and Learning 	
			 Rate = 0.1 since the model converged with a Learning
			 Rate of 0.01

	Convergence to the solution is observed with the fewest epochs, 10,
	using a learning rate of 0.1. 
	
	Convergence to the solution is not observed using 50 epochs and a 
	training rate of  0.01; displaying how critical its definition is. 

	A very nice observation is the results collapsing to smaller values 
	and losing their bias as an ideal combination of epochs and learning 
	rates are used. It would be interesting to force an overfit during 
	training and observe the residuals 

	