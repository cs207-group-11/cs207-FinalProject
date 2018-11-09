# We define a series of basic mathematics functions in this file
# that only involve ONE operand (e.g. log(e-base), sqrt etc).
import numpy as np
from AutoDiff import DualNumber

def log(x):
	"""Return the result of log.

	INPUTS
		x (DualNumber object or real number)

	RETURNS
		if x is a DualNumber, then return a DualNumber with val and der.
		if x is a real number, then return the value of log(x). 

	EXAMPLES
	>>> x = DualNumber(1)
	>>> t = log(x)
	>>> print(t.val, t.der)
	0.0 1.0
 	"""    
	try:
		return DualNumber(np.log(x.val), x.der / (x.val) )
	except AttributeError:
		return np.log(x)

def exp(x):
	"""Return the result of exp.

	INPUTS
		x (DualNumber object or real number)

	RETURNS
		if x is a DualNumber, then return a DualNumber with val and der.
		if x is a real number, then return the value of e^x.

	EXAMPLES
	>>> x = DualNumber(0)
	>>> t = exp(x)
	>>> print(t.val, t.der)
	1.0 1.0
 	"""    
	try:
		return DualNumber(np.exp(x.val), x.der * np.exp(x.val))
	except AttributeError:
		return np.exp(x)

def sqrt(x):
	"""Return the square root.

	INPUTS
		x (DualNumber object or real number)

	RETURNS
		if x is a DualNumber, then return a DualNumber with val and der.
		if x is a real number, then return the the square root of x.

	EXAMPLES
	>>> x = DualNumber(4)
	>>> t = sqrt(x)
	>>> print(t.val, t.der)
	2.0 0.25
 	"""    
	try:
		return DualNumber(np.sqrt(x.val), 0.5 * (x.val) ** (-0.5) * (x.der))
	except AttributeError:
		return np.sqrt(x)

def sin(x):
	"""Return the sin.

	INPUTS
		x (DualNumber object or real number)

	RETURNS
		if x is a DualNumber, then return a DualNumber with val and der.
		if x is a real number, then return the value of sin(x).

	EXAMPLES
	>>> x = DualNumber(0)
	>>> t = sin(x)
	>>> print(t.val, t.der)
	0.0 1.0
 	"""    
	try:
		return DualNumber(np.sin(x.val), np.cos(x.val) * (x.der))
	except AttributeError:
		return np.sin(x)

def cos(x):
	"""Return the cos.

	INPUTS
		x (DualNumber object or real number)

	RETURNS
		if x is a DualNumber, then return a DualNumber with val and der.
		if x is a real number, then return the value of cos(x).

	EXAMPLES
	>>> x = DualNumber(0)
	>>> t = cos(x)
	>>> print(t.val, t.der)
	1.0 -0.0
 	"""    
	try:
		return DualNumber(np.cos(x.val), -np.sin(x.val) * (x.der))
	except AttributeError:
		return np.cos(x)

def tan(x):
	"""Return the tan.

	INPUTS
		x (DualNumber object or real number)

	RETURNS
		if x is a DualNumber, then return a DualNumber with val and der.
		if x is a real number, then return the value of tan(x).

	EXAMPLES
	>>> x = DualNumber(0)
	>>> t = tan(x)
	>>> print(t.val, t.der)
	0.0 1.0
 	"""   
	try:
		return DualNumber(np.tan(x.val), 1/(np.cos(x.val)**2) * (x.der))
	except AttributeError:
		return np.tan(x)

def asin(x):
	"""Return the asin.

	INPUTS
		x (DualNumber object or real number)

	RETURNS
		if x is a DualNumber, then return a DualNumber with val and der.
		if x is a real number, then return the value of asin(x).

	EXAMPLES
	>>> x = DualNumber(0)
	>>> t = asin(x)
	>>> print(t.val, t.der)
	0.0 1.0
 	"""   
	try:
		return DualNumber(np.arcsin(x.val), 1/((1 - x.val**2)**0.5) * (x.der))
	except AttributeError:
		return np.arcsin(x)

def acos(x):
	"""Return the acos.

	INPUTS
		x (DualNumber object or real number)

	RETURNS
		if x is a DualNumber, then return a DualNumber with val and der.
		if x is a real number, then return the value of acos(x).

	EXAMPLES
	>>> x = DualNumber(0)
	>>> t = acos(x)
	>>> print(t.val, t.der)
	1.5707963267948966 -1.0
 	"""   
	try:
		return DualNumber(np.arccos(x.val), -1/((1 - x.val**2)**0.5) * (x.der))
	except AttributeError:
		return np.arccos(x)

def atan(x):
	"""Return the atan.

	INPUTS
		x (DualNumber object or real number)

	RETURNS
		if x is a DualNumber, then return a DualNumber with val and der.
		if x is a real number, then return the value of atan(x).

	EXAMPLES
	>>> x = DualNumber(0)
	>>> t = atan(x)
	>>> print(t.val, t.der)
	0.0 1.0
 	"""   	
	try:
		return DualNumber(np.arctan(x.val), 1/(1 + x.val**2) * (x.der))
	except AttributeError:
		return np.arctan(x)
