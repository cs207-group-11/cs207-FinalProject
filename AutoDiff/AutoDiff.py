import numpy as np
from collections import defaultdict

class Vector:
	def __init__(self, val=0.0, der=1.0, name=None):
		self.val = val
		if name:
			self.der = defaultdict(float)
			self.der[name] = der
		else:
			self.der = der

	def __add__(self, other):
		try:
			val = self.val + other.val
			ders = defaultdict(float)
			for key in self.der:
				ders[key] += self.der[key]
			for key in other.der:
				ders[key] += other.der[key]
			return Vector(val, ders)
		except AttributeError:
			return Vector(self.val + other, self.der)

	def __radd__(self, other):
		return self + other

	def __mul__(self, other):
		try:
			val = self.val * other.val
			ders = defaultdict(float)
			for key in self.der:
				ders[key] += self.der[key]*other.val
			for key in other.der:
				ders[key] += other.der[key]*self.val
			return Vector(val, ders)
		except AttributeError:
			ders = defaultdict(float)
			for key in self.der:
				ders[key] += self.der[key]*other
			return Vector(self.val * other, ders)

	def __rmul__(self, other):
		return self * other

	def __sub__(self, other):
		return self + (-1)*other

	def __rsub__(self, other):
		return other + (-1)*self

	def __pow__(self, other):
		try:
			val = self.val ** other.val
			ders = defaultdict(float)
			for key in self.der:
				ders[key] += other.val * self.val ** (other.val - 1) * self.der[key]
			for key in other.der:
				ders[key] += (self.val ** other.val) * np.log(self.val) * other.der[key]
			return Vector(val, ders)
		except AttributeError:
			val = self.val ** other
			ders = defaultdict(float)
			for key in self.der:
				ders[key] += other * (self.val ** (other-1)) * self.der[key]
			return Vector(val, ders)

	def __rpow__(self, other):
		val = other ** self.val
		ders = defaultdict(float)
		for key in self.der:
			ders[key] = other ** self.val * np.log(other) * self.der[key]
		return Vector(val, ders)

	def __truediv__(self, other):
		return self * (other ** (-1))

	def __rtruediv__(self, other):
		return other * (self ** (-1))

	def __neg__(self):
		ders = defaultdict(float)
		for key in self.der:
			ders[key] = -self.der[key]
		return Vector(-self.val, ders)

	def __pos__(self):
		return Vector(self.val, self.der)


class DualNumber:
	""" This class defines dual number and the way of a series of arithmetic functions and unary operations implemented on it."""
	def __init__(self, val = None, der = None):
		"""The constructor for DualNumber Class.
			For now this only applies for the case of scalar.

		Args:
			val (real number): The value of the function at a certain stage.
			der (real number): The value of the derivate at a ceratin stage. If no value given for der in dual number, the default is 1.
		"""
		if der is None:
			self.val = val
			self.der = 1
		else:
			self.val = val
			self.der = der

	def __add__(self, other):
		"""Return the result of self + other as a dual number.

		INPUTS
			self (DualNumber object): the recent DualNumber, the operand before '+'.
			other (DualNumber object or real number): the operand after '+'.

		RETURNS
			The result of self + other (DualNumber)

		EXAMPLES
		>>> x = DualNumber(3)
		>>> t = x + 2
		>>> print(t.val, t.der)
		5 1
 		"""
		try:
			return DualNumber(self.val + other.val, self.der + other.der)
		except AttributeError:
			return DualNumber(self.val + other, self.der)

	def __radd__(self, other):
		"""Return the result of other + self as a dual number using the __add__ above.

		INPUTS
			self (DualNumber object): the recent DualNumber, the operand after '+'.
			other (DualNumber object or real number): the operand before '+'.

		RETURNS
			The result of other + self (DualNumber)

		EXAMPLES
		>>> x = DualNumber(3)
		>>> t = 2 + x
		>>> print(t.val, t.der)
		5 1
		"""
		return self + other

	def __mul__(self, other):
		"""Return the result of self * other as a dual number.

		INPUTS
			self (DualNumber object): the recent DualNumber, the operand before '*'.
			other (DualNumber object or real number): the operand after '*'.

		RETURNS
			The result of self * other (DualNumber)

		EXAMPLES
		>>> x = DualNumber(3)
		>>> t = x * 2
		>>> print(t.val, t.der)
		6 2
		"""
		try:
			return DualNumber(self.val * other.val, self.der * other.val + self.val * other.der)
		except AttributeError:
			return DualNumber(self.val * other, self.der * other)

	def __rmul__(self, other):
		"""Return the result of other * self as a dual number using the __mul__ above.

		INPUTS
			self (DualNumber object): the recent DualNumber, the operand after '*'.
			other (DualNumber object or real number): the operand before '*'.

		RETURNS
			The result of other * self (DualNumber)

		EXAMPLES
		>>> x = DualNumber(3)
		>>> t = 2 * x
		>>> print(t.val, t.der)
		6 2
		"""
		return self * other

	def __sub__(self, other):
		"""Return the result of self - other as a dual number using the functions above.

		INPUTS
			self (DualNumber object): the recent DualNumber, the operand before '-'.
			other (DualNumber object or real number): the operand after '-'.

		RETURNS
			The result of self - other (DualNumber)

		EXAMPLES
		>>> x = DualNumber(1)
		>>> t = x-2
		>>> x = 1
		>>> print(t.val, t.der)
		-1 1
		"""
		return self + (-1) * other

	def __rsub__(self, other):
		"""Return the result of other - self as a dual number using the functions above.

		INPUTS
			self (DualNumber object): the recent DualNumber, the operand after '-'.
			other (DualNumber object or real number): the operand before '-'.

		RETURNS
			The result of other - self (DualNumber)

		EXAMPLES
		>>> x = DualNumber(1)
		>>> t = 2-x
		>>> print(t.val, t.der)
		1 -1
		"""
		return (-1) * self + other

	def __truediv__(self, other):
		"""Return the result of self/other as a dual number using other functions. (Python 3)

		INPUTS
			self (DualNumber object): the recent DualNumber, the operand before '/'.
			other (DualNumber object or real number): the operand after '/'.

		RETURNS
			The result of self/other (DualNumber)

		EXAMPLES
		>>> x = DualNumber(2)
		>>> t = x/2
		>>> print(t.val, t.der)
		1.0 0.5
		"""
		return self * (other ** (-1))

	def __rtruediv__(self, other):
		"""Return the result of other/self as a dual number using other functions. (Python 3)

		INPUTS
			self (DualNumber object): the recent DualNumber, the operand after '/'.
			other (DualNumber object or real number): the operand before '/'.

		RETURNS
			The result of other/self (DualNumber)

		EXAMPLES
		>>> x = DualNumber(2)
		>>> t = 2/x
		>>> print(t.val, t.der)
		1.0 -0.5
		"""
		return other*(self ** (-1))

	def __pow__(self, other):
		"""Return the result of self**(other) as a dual number using the functions above.

		INPUTS
			self (DualNumber object): the recent DualNumber, the base of '**'.
			other (DualNumber object or real number): the exponent of '**'.

		RETURNS
			The result of self**(other) (DualNumber)

		EXAMPLES
		>>> x = DualNumber(1)
		>>> t = x**2
		>>> print(t.val, t.der)
		1 2
		"""
		try:
			return DualNumber(self.val ** other.val, \
							 other.val * self.val ** (other.val - 1) * self.der + \
							 (self.val ** other.val) * np.log(self.val) * other.der)
		except AttributeError:
			return DualNumber(self.val ** other, other * self.val ** (other - 1) * self.der)

	def __rpow__(self, other):
		"""Return the result of ohter**(self) as a dual number using the functions above.

		INPUTS
			self (DualNumber object): the recent DualNumber, the exponent of '**'.
			other (DualNumber object or real number): the base of '**'.

		RETURNS
			The result of other**(self) (DualNumber)

		EXAMPLES
		>>> x = DualNumber(1)
		>>> t = 2**x
		>>> print(t.val, t.der)
		2 1.3862943611198906
		"""
		return DualNumber(other ** self.val, other ** self.val * np.log(other) * self.der)

	def __neg__(self):
		"""Return the result of negative unary operation (-self).

		INPUTS
			self (DualNumber object).

		RETURNS
			The result of (-self) (DualNumber)

		EXAMPLES
		>>> x = DualNumber(1)
		>>> t = 2+(-x)
		>>> print(t.val, t.der)
		1 -1
		"""
		return DualNumber(-self.val, -self.der)

	def __pos__(self):
		"""Return the result of positive unary operation (+self).

		INPUTS
			self (DualNumber object).

		RETURNS
			The result of (+self) (DualNumber)

		EXAMPLES
		>>> x = DualNumber(1)
		>>> t = 2+(+x)
		>>> print(t.val, t.der)
		3 1
		"""
		return DualNumber(self.val, self.der)


class ad:
	"""This class defines the object that the user will interact with and acts as a wrapper of the underlying DualNumber class"""
	def __init__(self):
		"""The constructor for ad Class."""
		pass

	def auto_diff(self, function, eval_point, order = 1):
		"""Return the value and derivative of the given founction at given point as a dual number.
		For now, it only stands for 1st order derivative.

		INPUTS
			self (ad object)
			function (function): the function defined by user
			eval_point (a real number): the point which the derivative will be computed at.
			order (real number): the order of derivative that the user want to compute, default = 1.

		RETURNS
			The value and derivative (DualNumber)

		EXAMPLES
		>>> ad = ad()
		>>> x = 1
		>>> user_def = lambda x: x ** x + 1 / x
		>>> t = ad.auto_diff(function = user_def, eval_point = x)
		>>> print(t.val, t.der)
		2.0 0.0
 		"""
		dual = DualNumber(eval_point)
		return function(dual)

	def auto_diff_vector(self, function, eval_point, order=1):
		return function(*eval_point)

if __name__ == '__main__':
	"""This part runs the doctest"""
	import doctest
	doctest.testmod()
