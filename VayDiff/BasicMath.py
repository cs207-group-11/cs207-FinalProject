import numpy as np
from VayDiff.VayDiff import Variable
from collections import defaultdict

def log(x):
	"""Return the result of log.

	INPUTS
		x (Variable object or real number)

	RETURNS
		if x is a Variable, then return a Variable with val and der for that single variable.
		if x is a real number, then return the value of np.log(x).

	EXAMPLES
	>>> x = Variable(1, name='x')
	>>> t = log(x)
	>>> print(t.val, t.der['x'])
	0.0 1.0
 	"""
	try:
		val = np.log(x.val)
		ders = defaultdict(float)
		for key in x.der:
			ders[key] += x.der[key]/x.val
		return Variable(val, ders)
	except AttributeError:
		return np.log(x)

def exp(x):
	"""Return the result of exp.

	INPUTS
		x (Variable object or real number)

	RETURNS
		if x is a Variable, then return a Variable with val and der.
		if x is a real number, then return the value of np.exp(x).

	EXAMPLES
	>>> x = Variable(0, name='x')
	>>> t = exp(x)
	>>> print(t.val, t.der['x'])
	1.0 1.0
 	"""
	try:
		val = np.exp(x.val)
		ders = defaultdict(float)
		for key in x.der:
			ders[key] += x.der[key] * np.exp(x.val)
		return Variable(val, ders)
	except AttributeError:
		return np.exp(x)

def sqrt(x):
	"""Return the square root.

	INPUTS
		x (Variable object or real number)

	RETURNS
		if x is a Variable, then return a Variable with val and der.
		if x is a real number, then return the the square root of x, np.sqrt(x).

	EXAMPLES
	>>> x = Variable(4, name='x')
	>>> t = sqrt(x)
	>>> print(t.val, t.der['x'])
	2.0 0.25
 	"""
	try:
		val = np.sqrt(x.val)
		ders = defaultdict(float)
		for key in x.der:
			ders[key] += 0.5 * (x.val ** (-0.5)) * (x.der[key])
		return Variable(val, ders)
	except AttributeError:
		return np.sqrt(x)

def sin(x):
	"""Return the sine.

	INPUTS
		x (Variable object or real number)

	RETURNS
		if x is a Variable, then return a Variable with val and der.
		if x is a real number, then return the value of np.sin(x).

	EXAMPLES
	>>> x = Variable(0, name='x')
	>>> t = sin(x)
	>>> print(t.val, t.der['x'])
	0.0 1.0
 	"""
	try:
		val = np.sin(x.val)
		ders = defaultdict(float)
		for key in x.der:
			ders[key] += np.cos(x.val) * (x.der[key])
		return Variable(val, ders)
	except AttributeError:
		return np.sin(x)

def cos(x):
	"""Return the cosine.

	INPUTS
		x (Variable object or real number)

	RETURNS
		if x is a Variable, then return a Variable with val and der.
		if x is a real number, then return the value of np.cos(x).

	EXAMPLES
	>>> x = Variable(0, name='x')
	>>> t = cos(x)
	>>> print(t.val, t.der['x'])
	1.0 0.0
 	"""
	try:
		val = np.cos(x.val)
		ders = defaultdict(float)
		for key in x.der:
			ders[key] += -np.sin(x.val) * (x.der[key])
		return Variable(val, ders)
	except AttributeError:
		return np.cos(x)

def tan(x):
	"""Return the tangent.

	INPUTS
		x (Variable object or real number)

	RETURNS
		if x is a Variable, then return a Variable with val and der.
		if x is a real number, then return the value of np.tan(x).

	EXAMPLES
	>>> x = Variable(0, name='x')
	>>> t = tan(x)
	>>> print(t.val, t.der['x'])
	0.0 1.0
 	"""
	try:
		val = np.tan(x.val)
		ders = defaultdict(float)
		for key in x.der:
			ders[key] += 1/(np.cos(x.val)**2) * (x.der[key])
		return Variable(val, ders)
	except AttributeError:
		return np.tan(x)

def arcsin(x):
	"""Return the inverse sine or the arcsin.

	INPUTS
		x (Variable object or real number)

	RETURNS
		if x is a Variable, then return a Variable with val and der.
		if x is a real number, then return the value of arcsin(x).

	EXAMPLES
	>>> x = Variable(0, name='x')
	>>> t = arcsin(x)
	>>> print(t.val, t.der['x'])
	0.0 1.0
 	"""
	try:
		val = np.arcsin(x.val)
		ders = defaultdict(float)
		for key in x.der:
			ders[key] += 1/((1 - x.val**2)**0.5) * (x.der[key])
		return Variable(val, ders)
	except AttributeError:
		return np.arcsin(x)

def arccos(x):
	"""Return the inverse cosine or the arccos.

	INPUTS
		x (Variable object or real number)

	RETURNS
		if x is a Variable, then return a Variable with val and der.
		if x is a real number, then return the value of arccos(x).

	EXAMPLES
	>>> x = Variable(0, name='x')
	>>> t = arccos(x)
	>>> print(t.val, t.der['x'])
	1.5707963267948966 -1.0
 	"""
	try:
		val = np.arccos(x.val)
		ders = defaultdict(float)
		for key in x.der:
			ders[key] += -1/((1 - x.val**2)**0.5) * (x.der[key])
		return Variable(val, ders)
	except AttributeError:
		return np.arccos(x)

def arctan(x):
	"""Return the inverse tangent or the arctan.

	INPUTS
		x (Variable object or real number)

	RETURNS
		if x is a Variable, then return a Variable with val and der.
		if x is a real number, then return the value of arctan(x).

	EXAMPLES
	>>> x = Variable(0, name='x')
	>>> t = arctan(x)
	>>> print(t.val, t.der['x'])
	0.0 1.0
 	"""
	try:
		val = np.arctan(x.val)
		ders = defaultdict(float)
		for key in x.der:
			ders[key] += 1/(1 + x.val**2) * (x.der[key])
		return Variable(val, ders)
	except AttributeError:
		return np.arctan(x)

def sinh(x):
	"""The hyperbolic sine or the sinh

	INPUTS
		x (Variable object or real number)

	RETURNS
		if x is a Variable, then return a Variable with val and der.
		if x is a real number, then return the value of np.sinh(x).

	EXAMPLES
	>>> x = Variable(0, name='x')
	>>> t = sinh(x)
	>>> print(t.val, t.der['x'])
	0.0 1.0
	"""
	try:
		val = np.sinh(x.val)
		ders = defaultdict(float)
		for key in x.der:
			ders[key] += np.cosh(x.val) * (x.der[key])
		return Variable(val, ders)
	except AttributeError:
		return np.sinh(x)

def cosh(x):
	"""The hyperbolic cosine or the cosh

	INPUTS
		x (Variable object or real number)

	RETURNS
		if x is a Variable, then return a Variable with val and der.
		if x is a real number, then return the value of np.cosh(x).

	EXAMPLES
	>>> x = Variable(0, name='x')
	>>> t = cosh(x)
	>>> print(t.val, t.der['x'])
	1.0 0.0
	"""
	try:
		val = np.cosh(x.val)
		ders = defaultdict(float)
		for key in x.der:
			ders[key] += np.sinh(x.val) * (x.der[key])
		return Variable(val, ders)
	except AttributeError:
		return np.cosh(x)

def tanh(x):
	"""The hyperbolic tangent or the tanh

	INPUTS
		x (Variable object or real number)

	RETURNS
		if x is a Variable, then return a Variable with val and der.
		if x is a real number, then return the value of np.tanh(x).

	EXAMPLES
	>>> x = Variable(0, name='x')
	>>> t = tanh(x)
	>>> print(t.val, t.der['x'])
	0.0 1.0
	"""
	try:
		val = np.tanh(x.val)
		ders = defaultdict(float)
		for key in x.der:
			ders[key] += (1/(np.cosh(x.val))**2) * (x.der[key])
		return Variable(val, ders)
	except AttributeError:
		return np.tanh(x)

if __name__ == '__main__':
	"""This part runs the doctest"""
	import doctest
	doctest.testmod()
