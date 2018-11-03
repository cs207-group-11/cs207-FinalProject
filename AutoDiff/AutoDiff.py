import numpy as np
## haven't figured out a good way to import this yet
from . import BasicMath as bm

class DualNumber():
	def __init__(self, val = None, der = None):
		# This only applies for the case of scalar
		if der is None:
			self.val = val
			self.der = 1
		else:
			self.val = val
			self.der = der

	def __add__(self, other):
		if isinstance(other, DualNumber):
			return DualNumber(self.val + other.val, self.der + other.der)
		else:
			return DualNumber(self.val + other, self.der)

	def __radd__(self, other):
		# use __add__ defined above
		return self + other

	def __mul__(self, other):
		if isinstance(other, DualNumber):
			return DualNumber(self.val * other.val, self.der * other.val + self.val * other.der)
		else:
			return DualNumber(self.val * other, self.der * other)

	def __rmul__(self, other):
		return self * other

	def __sub__(self, other):
		# Take advantage of the addition defined above
		return self + (-1) * other

	def __rsub__(self, other):
		return (-1) * self + other

	def __div__(self, other):
		return self * (other ** (-1))

	def __rdiv__(self, other):
		return other*(self ** (-1))

	def __pow__(self, other):
		if isinstance(other, DualNumber):
			return DualNumber(self.val ** other.val, \
							 other.val * self.val ** (other.val - 1) * self.der + \
							 (self.val ** other.val) * bm.log(self.val) * other.der)
		else:
			return DualNumber(self.val ** other, other * self.val ** (other - 1) * self.der)
	def __rpow__(self, other):
		return DualNumber(other ** self.val, other ** self.val * bm.log(other) * self.der)

def log(x):
	if isinstance(x, DualNumber):
		return DualNumber(np.log(x.val), x.der / (x.val) )
	else:
		return np.log(x)

def exp(x):
	if isinstance(x, DualNumber):
		return DualNumber(np.exp(x.val), x.der * np.exp(x.val))
	else:
		return np.exp(x)

def sqrt(x):
	if isinstance(x, DualNumber):
		return DualNumber(np.sqrt(x.val), 0.5 * (x.val) ** (-0.5) * (x.der))
	else:
		return np.sqrt(x)

def sin(x):
	if isinstance(x, DualNumber):
		return DualNumber(np.sin(x.val), np.cos(x.val) * (x.der))
	else:
		return np.sin(x)

def cos(x):
	if isinstance(x, DualNumber):
		return DualNumber(np.cos(x.val), -np.sin(x.val) * (x.der))
	else:
		return np.cos(x)

def tan(x):
	if isinstance(x, DualNumber):
		return DualNumber(np.tan(x.val), 1/(np.cos(x.val)**2) * (x.der))
	else:
		return np.tan(x)

def asin(x):
	if isinstance(x, DualNumber):
		return DualNumber(np.arcsin(x.val), 1/((1 - x.val**2)**0.5) * (x.der))
	else:
		return np.arcsin(x)

def acos(x):
	if isinstance(x, DualNumber):
		return DualNumber(np.arccos(x.val), -1/((1 - x.val**2)**0.5) * (x.der))
	else:
		return np.arccos(x)

def atan(x):
	if isinstance(x, DualNumber):
		return DualNumber(np.arctan(x.val), 1/(1 + x.val**2) * (x.der))
	else:
		return np.arctan(x)

class ad():
	"""
    >>> 1+1
    2
    """
	def __init__(self):
		pass
	def auto_diff(self, function, eval_point, order = 1):
		dual = DualNumber(eval_point)
		return function(dual)


# This is a user-defined function. Feel free to change this function to whatever function you are interested in
# make sure to use the function defined above
def user_defined(x):
	return sin(x) ** 2 + cos(x) * 1 / (x**2)

if __name__ == '__main__':
	import doctest
	doctest.testmod()
	ad = ad()
	x = 0.2
	t = ad.auto_diff(function = user_defined, eval_point = x)
	print (type(t))
	print (t.val)
	print (t.der)

# You can confirm the result using Wolfram Alpha!
