import numpy as np
def precision(y_true, y_pred):
	# Your code here
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	# precision = T.P/(T.P + F.P ) 
	t_p = 0
	f_p = 0
	for i in range(len(y_true)):
		if (y_true[i] == 1 and y_pred[i] == 1):
			t_p += 1
		if (y_true[i] == 0 and y_pred[i] == 1):
			f_p += 1
	precision = t_p /(t_p + f_p)
	return precision 


