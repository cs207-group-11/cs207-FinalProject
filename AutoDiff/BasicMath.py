# We define a series of basic mathematics functions in this file
# that only involve ONE operand (e.g. log(e-base), sqrt etc).
import numpy as np
from AutoDiff.AutoDiff import DualNumber

def log(x):
	try:
		return DualNumber(np.log(x.val), x.der / (x.val) )
	except AttributeError:
		return np.log(x)

def exp(x):
	try:
		return DualNumber(np.exp(x.val), x.der * np.exp(x.val))
	except AttributeError:
		return np.exp(x)

def sqrt(x):
	try:
		return DualNumber(np.sqrt(x.val), 0.5 * (x.val) ** (-0.5) * (x.der))
	except AttributeError:
		return np.sqrt(x)

def sin(x):
	try:
		return DualNumber(np.sin(x.val), np.cos(x.val) * (x.der))
	except AttributeError:
		return np.sin(x)

def cos(x):
	try:
		return DualNumber(np.cos(x.val), -np.sin(x.val) * (x.der))
	except AttributeError:
		return np.cos(x)

def tan(x):
	try:
		return DualNumber(np.tan(x.val), 1/(np.cos(x.val)**2) * (x.der))
	except AttributeError:
		return np.tan(x)

def asin(x):
	try:
		return DualNumber(np.arcsin(x.val), 1/((1 - x.val**2)**0.5) * (x.der))
	except AttributeError:
		return np.arcsin(x)

def acos(x):
	try:
		return DualNumber(np.arccos(x.val), -1/((1 - x.val**2)**0.5) * (x.der))
	except AttributeError:
		return np.arccos(x)

def atan(x):
	try:
		return DualNumber(np.arctan(x.val), 1/(1 + x.val**2) * (x.der))
	except AttributeError:
		return np.arctan(x)
