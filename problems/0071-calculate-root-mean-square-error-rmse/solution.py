
import numpy as np

def rmse(y_true, y_pred):
	if (y_true.shape != y_pred.shape):
		raise valueError("Shape must be same")
	if(y_true.shape == 0):
		raise valueError("size must not be zero")

	rmse_res = np.sqrt(np.mean((y_pred - y_true)**2))
	
	return round(rmse_res,3)
