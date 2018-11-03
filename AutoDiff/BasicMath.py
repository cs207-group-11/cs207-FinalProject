# We define a series of basic mathematics functions in this file
# that only involve ONE operand (e.g. log(e-base), sqrt etc). 
 
import numpy as np
def log(x):
	if isinstance(x, AutoDiff.DualNumber):
		pass
	return np.log(x)

def exp(x):
	return np.exp(x)

def sqrt(x):
	return np.sqrt(x)

def sin(x):
	return np.sin(x)

def cos(x):
	return np.cos(x)

def tan(x):
	return np.tan(x)

def asin(x):
	return np.arcsin(x)

def acos(x):
	return np.arccos(x)

def atan(x):
	return np.arctan(x)

