import numpy as np
from collections import defaultdict

class Variable:
	"""
	This class defines a variable as a Dual Number under the hood.
	A series of arithmetic functions and unary operations implemented on this variable are defined here.
	This is the elementary way by which a user can input a variable to be differentiated over in our VayDiff class.
	"""
	def __init__(self, val=0.0, der=1.0, sec_der=0.0, name=None):
		"""The constructor for Variable Class.

		Args:
			val (real number): The value of the variable. The function will be differentiated at this value.
			der (real number): The value of the derivative. Default is 1.
			name (string): The name of the variable. The default is None.
		"""
		self.val = val
		self.name = name
		if name:
			self.der = defaultdict(float)
			self.sec_der = defaultdict(float)
			self.der[name] = der
			self.sec_der[name] = sec_der
		else:
			self.der = der
			self.sec_der = sec_der

	def __add__(self, other):
		"""Return the result of self + other as a variable.

		INPUTS
			self (Variable object): the recent Variable, the operand before '+'.
			other (Variable object or real number): the operand after '+'.

		RETURNS
			The result of self + other (Variable)

		EXAMPLES
		>>> x = Variable(3, name='x')
		>>> t = x + 2
		>>> print(t.val, t.der['x'])
		5 1.0
 		"""
		try:
			val = self.val + other.val
			ders = defaultdict(float)
			sec_ders = defaultdict(float)
			for key in self.der:
				ders[key] += self.der[key]
				sec_ders[key] += self.sec_der[key]
			for key in other.der:
				ders[key] += other.der[key]
				sec_ders[key] += other.sec_der[key]
			return Variable(val, ders, sec_ders)
		except AttributeError:
			return Variable(self.val + other, self.der, self.sec_der)

	def __radd__(self, other):
		"""Return the result of other + self as a variable using the __add__ above.

		INPUTS
			self (Variable object): the recent Variable, the operand after '+'.
			other (Variable object or real number): the operand before '+'.

		RETURNS
			The result of other + self (Variable)

		EXAMPLES
		>>> x = Variable(3, name='x')
		>>> t = 2 + x
		>>> print(t.val, t.der['x'])
		5 1.0
		"""
		return self + other

	def __mul__(self, other):
		"""Return the result of self * other as a variable.

		INPUTS
			self (Variable object): the recent Variable, the operand before '*'.
			other (Variable object or real number): the operand after '*'.

		RETURNS
			The result of self * other (Variable)

		EXAMPLES
		>>> x = Variable(3, name='x')
		>>> t = x * 2
		>>> print(t.val, t.der['x'])
		6 2.0
		"""
		try:
			val = self.val * other.val
			ders = defaultdict(float)
			sec_ders = defaultdict(float)
			for key in self.der:
				ders[key] += self.der[key]*other.val
				sec_ders[key] += other.val*self.sec_der[key]+self.der[key]*other.der[key]
			for key in other.der:
				ders[key] += other.der[key]*self.val
				sec_ders[key] += self.val*other.sec_der[key]+other.der[key]*self.der[key]
			return Variable(val, ders, sec_ders)
		except AttributeError:
			ders = defaultdict(float)
			sec_ders = defaultdict(float)
			for key in self.der:
				ders[key] += self.der[key]*other
				sec_ders[key] += self.sec_der[key]*other
			return Variable(self.val * other, ders, sec_ders)

	def __rmul__(self, other):
		"""Return the result of other * self as a variable using the __mul__ above.

		INPUTS
			self (Variable object): the recent Variable, the operand after '*'.
			other (Variable object or real number): the operand before '*'.

		RETURNS
			The result of other * self (Variable)

		EXAMPLES
		>>> x = Variable(3, name='x')
		>>> t = 2 * x
		>>> print(t.val, t.der['x'])
		6 2.0
		"""
		return self * other

	def __sub__(self, other):
		"""Return the result of self - other as a variable using the functions above.

		INPUTS
			self (Variable object): the recent Variable, the operand before '-'.
			other (Variable object or real number): the operand after '-'.

		RETURNS
			The result of self - other (Variable)

		EXAMPLES
		>>> x = Variable(1, name='x')
		>>> t = x-2
		>>> x = 1
		>>> print(t.val, t.der['x'])
		-1 1.0
		"""
		return self + (-1)*other

	def __rsub__(self, other):
		"""Return the result of other - self as a variable using the functions above.

		INPUTS
			self (Variable object): the recent Variable, the operand after '-'.
			other (Variable object or real number): the operand before '-'.

		RETURNS
			The result of other - self (Variable)

		EXAMPLES
		>>> x = Variable(1, name='x')
		>>> t = 2-x
		>>> print(t.val, t.der['x'])
		1 -1.0
		"""
		return other + (-1)*self

	def __pow__(self, other):
		"""Return the result of self**(other) as a variable using the functions above.

		INPUTS
			self (Variable object): the recent Variable, the base of '**'.
			other (Variable object or real number): the exponent of '**'.

		RETURNS
			The result of self**(other) (Variable)

		EXAMPLES
		>>> x = Variable(1, name='x')
		>>> t = x**2
		>>> print(t.val, t.der['x'])
		1 2.0
		"""
		try:
			val = self.val ** other.val
			ders = defaultdict(float)
			sec_ders = defaultdict(float)
			for key in self.der:
				ders[key] += other.val * self.val ** (other.val - 1) * self.der[key]
				sec_ders[key] += other.val * (other.val -1) * self.val ** (other.val - 2) * self.der[key]
			for key in other.der:
				ders[key] += (val) * np.log(self.val) * other.der[key]
				sec_ders[key] += (val) * (np.log(self.val)**2) * other.der[key]
			return Variable(val, ders, sec_ders)
		except AttributeError:
			val = self.val ** other
			ders = defaultdict(float)
			sec_ders = defaultdict(float)
			for key in self.der:
				ders[key] += other * (self.val ** (other-1)) * self.der[key]
				sec_ders[key] += other * (other -1) * (self.val ** (other-2)) * self.der[key]
			return Variable(val, ders, sec_ders)

	def __rpow__(self, other):
		"""Return the result of other**(self) as a variable using the functions above.

		INPUTS
			self (Variable object): the recent Variable, the exponent of '**'.
			other (Variable object or real number): the base of '**'.

		RETURNS
			The result of other**(self) (Variable)

		EXAMPLES
		>>> x = Variable(1, name='x')
		>>> t = 2**x
		>>> print(t.val, t.der['x'])
		2 1.3862943611198906
		"""
		val = other ** self.val
		ders = defaultdict(float)
		sec_ders = defaultdict(float)
		for key in self.der:
			ders[key] = val * np.log(other) * self.der[key]
			sec_ders[key] = (val) * (np.log(other)**2) * self.der[key]
		return Variable(val, ders, sec_ders)

	def __truediv__(self, other):
		"""Return the result of self/other as a variable using other functions. (Python 3)

		INPUTS
			self (Variable object): the recent Variable, the operand before '/'.
			other (Variable object or real number): the operand after '/'.

		RETURNS
			The result of self/other (Variable)

		EXAMPLES
		>>> x = Variable(2, name='x')
		>>> t = x/2
		>>> print(t.val, t.der['x'])
		1.0 0.5
		"""
		return self * (other ** (-1))

	def __rtruediv__(self, other):
		"""Return the result of other/self as a variable using other functions. (Python 3)

		INPUTS
			self (Variable object): the recent Variable, the operand after '/'.
			other (Variable object or real number): the operand before '/'.

		RETURNS
			The result of other/self (Variable)

		EXAMPLES
		>>> x = Variable(2, name='x')
		>>> t = 2/x
		>>> print(t.val, t.der['x'])
		1.0 -0.5
		"""
		return other * (self ** (-1))

	def __neg__(self):
		"""Return the result of negative unary operation (-self).

		INPUTS
			self (Variable object).

		RETURNS
			The result of (-self) (Variable)

		EXAMPLES
		>>> x = Variable(1, name='x')
		>>> t = 2+(-x)
		>>> print(t.val, t.der['x'])
		1 -1.0
		"""
		ders = defaultdict(float)
		sec_ders = defaultdict(float)
		for key in self.der:
			ders[key] = -self.der[key]
			sec_ders[key] = -self.sec_der[key]
		return Variable(-self.val, ders, sec_ders)

	def __pos__(self):
		"""Return the result of positive unary operation (+self).

		INPUTS
			self (Variable object).

		RETURNS
			The result of (+self) (Variable)

		EXAMPLES
		>>> x = Variable(1, name='x')
		>>> t = 2+(+x)
		>>> print(t.val, t.der['x'])
		3 1.0
		"""
		return Variable(self.val, self.der, self.sec_der)

	def __eq__(self,other):
		"""Return the result of (equal to) comparison.

		INPUTS
			self (Variable object or real number): the operand before '=='
			other (Variable object or real number): the operand after '=='

		RETURNS
			The result of the comparison (boolean)

		EXAMPLES
		>>> x = Variable(1,2,name='x')
		>>> y = Variable(1,2,name='y')
		>>> x == y
		True
		"""

		try:
			return (self.val == other.val) and (list(self.der.values()) == list(other.der.values()))
		except AttributeError:
			return False

	def __ne__(self,other):
		"""Return the result of (not equal to) comparison.

		INPUTS
			self (Variable object or real number): the operand before '!='
			other (Variable object or real number): the operand after '!='

		RETURNS
			The result of the comparison (boolean)

		EXAMPLES
		>>> x = Variable(1,2,name='x')
		>>> y = Variable(1,2,name='y')
		>>> x != y
		False
		"""
		try:
			return (self.val != other.val) or (list(self.der.values()) != list(other.der.values()))
		except AttributeError:
			return True

class Diff:
	"""This class defines the object that the user will interact with and acts as a wrapper of the underlying Variable class"""
	def __init__(self):
		"""The constructor for Diff Class."""
		pass

	def auto_diff(self, function, eval_point):
		"""Return the value and derivative of the given founction at given point as a variable.
		For now, it only stands for 1st order derivative.

		INPUTS
			self (Diff object)
			function (function): the function defined by user
			eval_points (a list of Variable objects): the point(s) which the derivative will be computed at.

		RETURNS
			The value and derivative (Variable)

		EXAMPLES
		>>> ad = Diff()
		>>> x = Variable(1, name='x')
		>>> user_def = lambda x: x ** x + 1 / x
		>>> t = ad.auto_diff(function = user_def, eval_point = [x])
		>>> print(t.val, t.der['x'])
		2.0 0.0
		>>> y = Variable(2, name='y')
		>>> user_def_xy = lambda x,y: x+2*y
		>>> t = ad.auto_diff(function = user_def_xy, eval_point = [x,y])
		>>> print(t.val, t.der['x'])
		5 1.0
		>>> print(t.val, t.der['y'])
		5 2.0
 		"""
		return function(*eval_point)

	def jacobian(self, functions, eval_points):
		"""Return the Jacobian of a list of functions.

		INPUTS
			self (Diff object)
			functions: a list of functions defined by user
			eval_points (a list of Variable objects): the variables which the derivative will be computed at.

		RETURNS
			A Jacobian Matrix, a n by p Numpy array where n is the number of functions input by the user and p is the
			number of variables to differentiate over.

		EXAMPLES
		>>> f1 = lambda x,y: x**2*y
		>>> f2 = lambda x,y: 5*y+x
		>>> x = Variable(val=3, name='x')
	    >>> y = Variable(val=5, name='y')
	    >>> t1 = Diff().jacobian([f1,f2], [x,y])
		>>> t1.shape
		(2, 2)
		>>> t1[0]
		array([30.,  9.])
		>>> t1[1]
		array([1., 5.])
 		"""
		key_order = [k.name for k in eval_points]
		output = np.ones(shape=(len(functions), len(eval_points)))
		for i, func in enumerate(functions):
			eval = self.auto_diff(func, eval_points)
			ders = eval.der
			for j, k in enumerate(key_order):
				output[i][j] = ders[k]
		return output

	def hessian(self, functions, eval_points):
		"""Return the Hessian of a list of functions."""
		raise NotImplementedError

if __name__ == '__main__':
	"""This part runs the doctest"""
	import doctest
	doctest.testmod()
