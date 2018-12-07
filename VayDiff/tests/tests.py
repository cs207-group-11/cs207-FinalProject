import pytest
import numpy as np
from VayDiff import BasicMath as bm
from VayDiff.VayDiff import Diff
from VayDiff.VayDiff import Variable

epsilon = 1e-7

def test_init():
    # test Variable __init__
    # test BasicMath __init__
    return

# Simple operations
def add_function(x):
    return 2+x+x-3

def sub_function(x):
    return 3-x

def mul_function(x):
    return 7 + 3.3*x + x*2.7 + 4

def div_scalar(x):
    return x/3

def div_function(x):
    return x/3 + 6.0/x - 1

def simple_op_complete(x):
    return x*x + x/x + 3*x*2 - 3*(x+2)

def unary(x):
    return 3*(+x) + (-x)

# Testing addition, subtraction, multiplication and division
def test_simple_operators():
    a = Variable(val=3, name='a')
    # Addition and subtraction
    t11 = Diff().auto_diff(function = add_function, eval_point = [a])
    t12 = Diff().auto_diff(function = sub_function, eval_point = [a])
    assert(t11.val == 5 and t11.der['a'] == 2)
    assert(t12.val == 0 and t12.der['a'] == -1)

    # Multiplication
    t2 = Diff().auto_diff(function = mul_function, eval_point = [a])
    assert(t2.val == 29)
    assert(t2.der['a'] == 6)

    # Division
    t3 = Diff().auto_diff(function = div_scalar, eval_point = [a])
    assert(t3.val == 1)
    assert(t3.der['a'] == 1/3)

    t4 = Diff().auto_diff(function = div_function, eval_point = [a])
    assert(t4.val == 2)
    assert(t4.der['a'] == -1/3)

    t5 = Diff().auto_diff(function = simple_op_complete, eval_point = [a])
    assert(t5.val == 13)
    assert(t5.der['a'] == 9)

    # Unary
    t6 = Diff().auto_diff(function = unary, eval_point = [a])
    assert(t6.val == 6)
    assert(t6.der['a'] == 2)

# More advanced operators
def power_scalar(x):
    return 10+x**2

def sqrt_scalar(x):
    return 10 + bm.sqrt(2*x+1)

def power_function(x):
    return 2**x

def power_function_general(x):
    return x**x

def test_more_operators():
    a = Variable(val=4, name='a')
    # Power
    t1 = Diff().auto_diff(function = power_scalar, eval_point = [a])
    assert(t1.val == 26)
    assert(t1.der['a'] == 8)

    # Square root
    t2 = Diff().auto_diff(function = sqrt_scalar, eval_point = [a])
    assert(t2.val == 13)
    assert(t2.der['a'] == 1/3)

    # Power function
    t3 = Diff().auto_diff(function = power_function, eval_point = [a])
    assert(t3.val == 16)
    assert(t3.der['a'] == bm.log(2)*16)
    assert(t3.der['a'] == np.log(2)*16)

    # Power function general
    t4 = Diff().auto_diff(function = power_function_general, eval_point = [a])
    assert(t4.val == 256)
    assert(t4.der['a'] == 256*(bm.log(4)+1))
    assert(t4.der['a'] == 256*(np.log(4)+1))

# Trig Functions
def sin_cos(x):
    return bm.sin(x) + bm.cos(x)

def tan_function(x):
    return bm.tan(2*x)

def inverse_trig(x):
    # arcsin, arccos, arctan
    return bm.arcsin(x) - bm.arccos(x) + bm.arctan(x)

def test_trig():
    a = Variable(val=0, name='a')
    # Sine and cosine
    t1 = Diff().auto_diff(function = sin_cos, eval_point = [a])
    assert(t1.val == 1)
    assert(t1.der['a'] == 1)

    # Tan
    t2 = Diff().auto_diff(function = tan_function, eval_point = [a])
    assert(t2.val == 0)
    assert(t2.der['a'] == 2)

    # Arcsin, arccos, arctan
    t3 = Diff().auto_diff(function = inverse_trig, eval_point = [a])
    assert(np.abs(t3.val + np.pi/2) < epsilon)
    assert(t3.der['a'] > 0)

# Other functions
def log_exp_function(x):
    # log, exp
    return bm.log(x) + bm.exp(x)

def test_other_functions():
    a = Variable(val=1, name='a')
    # log, exp
    t1 = Diff().auto_diff(function = log_exp_function, eval_point = [a])
    assert(t1.val == np.exp(1))
    assert(t1.der['a'] == np.exp(1) + 1)

# Functions that could be problematic
def div_zero(x):
    return x/0

def test_problematic():
    a = Variable(val=2, name='a')
    with pytest.raises(ZeroDivisionError):
        t1 = Diff().auto_diff(function = div_zero, eval_point = [a])

def test_sanity_checks():
    assert(bm.log(4.1) == np.log(4.1))
    assert(bm.exp(-10.1) == np.exp(-10.1))
    assert(bm.sqrt(112.3) == np.sqrt(112.3))
    assert(bm.sin(4.1) == np.sin(4.1))
    assert(bm.cos(2.2) == np.cos(2.2))
    assert(bm.tan(2) == np.tan(2.0))
    assert(bm.arcsin(0) == np.arcsin(0))
    assert(bm.arccos(0.2) == np.arccos(0.2))
    assert(bm.arctan(2) == np.arctan(2))
    assert(bm.sinh(2) == np.sinh(2))
    assert(bm.cosh(2) == np.cosh(2))
    assert(bm.tanh(2) == np.tanh(2))

test_init()
test_simple_operators()
test_more_operators()
test_trig()
test_other_functions()
test_problematic()
test_sanity_checks()
