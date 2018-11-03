import numpy as np
import BasicMath as bm


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


class ad():
	def __init__(self):
		pass
	def auto_diff(self, function, eval_point, order = 1):
		dual = DualNumber(eval_point)
		return function(dual)



def multi(x):
	return x ** 2


ad = ad()
x = 3
t = ad.auto_diff(function = multi, eval_point = x)
print (t.val)
print (t.der)
