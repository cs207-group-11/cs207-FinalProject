import pytest
import numpy as np
from AutoDiff import AutoDiff as ad_full
from AutoDiff.AutoDiff import ad

epsilon = 1e-7

def test_init():
    # test DualNumber __init__
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
    a = 3
    # Addition and subtraction
    t11 = ad().auto_diff(function = add_function, eval_point = a)
    t12 = ad().auto_diff(function = sub_function, eval_point = a)
    assert(t11.val == 5 and t11.der == 2)
    assert(t12.val == 0 and t12.der == -1)

    # Multiplication
    t2 = ad().auto_diff(function = mul_function, eval_point = a)
    assert(t2.val == 29)
    assert(t2.der == 6)

    # Division
    t3 = ad().auto_diff(function = div_scalar, eval_point = a)
    assert(t3.val == 1)
    assert(t3.der == 1/3)

    t4 = ad().auto_diff(function = div_function, eval_point = a)
    assert(t4.val == 2)
    assert(t4.der == -1/3)

    t5 = ad().auto_diff(function = simple_op_complete, eval_point = a)
    assert(t5.val == 13)
    assert(t5.der == 9)

    # Unary
    t6 = ad().auto_diff(function = unary, eval_point = a)
    assert(t6.val == 6)
    assert(t6.der == 2)

# More advanced operators
def power_scalar(x):
    return 10+x**2

def sqrt_scalar(x):
    return 10 + ad_full.sqrt(2*x+1)

def power_function(x):
    return 2**x

def power_function_general(x):
    return x**x

def test_more_operators():
    a = 4
    # Power
    t1 = ad().auto_diff(function = power_scalar, eval_point = a)
    assert(t1.val == 26)
    assert(t1.der == 8)

    # Square root
    t2 = ad().auto_diff(function = sqrt_scalar, eval_point = a)
    assert(t2.val == 13)
    assert(t2.der == 1/3)

    # Power function
    t3 = ad().auto_diff(function = power_function, eval_point = a)
    assert(t3.val == 16)
    assert(t3.der == ad_full.log(2)*16)
    assert(t3.der == np.log(2)*16)

    # Power function general
    t4 = ad().auto_diff(function = power_function_general, eval_point = a)
    assert(t4.val == 256)
    assert(t4.der == 256*(ad_full.log(4)+1))
    assert(t4.der == 256*(np.log(4)+1))

# Trig Functions
def sin_cos(x):
    return ad_full.sin(x) + ad_full.cos(x)

def tan_function(x):
    return ad_full.tan(2*x)

def inverse_trig(x):
    # arcsin, arccos, arctan
    return ad_full.asin(x) - ad_full.acos(x) + ad_full.atan(x)

def test_trig():
    a = 0
    # Sine and cosine
    t1 = ad().auto_diff(function = sin_cos, eval_point = a)
    assert(t1.val == 1)
    assert(t1.der == 1)

    # Tan
    t2 = ad().auto_diff(function = tan_function, eval_point = a)
    assert(t2.val == 0)
    assert(t2.der == 2)

    # Arcsin, arccos, arctan
    t3 = ad().auto_diff(function = inverse_trig, eval_point = a)
    assert(np.abs(t3.val + np.pi/2) < epsilon)
    assert(t3.der > 0)

# Other functions
def log_exp_function(x):
    # log, exp
    return ad_full.log(x) + ad_full.exp(x)

def test_other_functions():
    a = 1
    # log, exp
    t1 = ad().auto_diff(function = log_exp_function, eval_point = a)
    assert(t1.val == np.exp(1))
    assert(t1.der == np.exp(1) + 1)

# Functions that could be problematic
def div_zero(x):
    return x/0

def test_problematic():
    a = 2
    with pytest.raises(ZeroDivisionError):
        t1 = ad().auto_diff(function = div_zero, eval_point = a)

def test_sanity_checks():
    assert(ad_full.log(4.1) == np.log(4.1))
    assert(ad_full.exp(-10.1) == np.exp(-10.1))
    assert(ad_full.sqrt(112.3) == np.sqrt(112.3))
    assert(ad_full.sin(4.1) == np.sin(4.1))
    assert(ad_full.cos(2.2) == np.cos(2.2))
    assert(ad_full.tan(2) == np.tan(2.0))
    assert(ad_full.asin(0) == np.arcsin(0))
    assert(ad_full.acos(0.2) == np.arccos(0.2))
    assert(ad_full.atan(2) == np.arctan(2))

test_init()
test_simple_operators()
test_more_operators()
test_trig()
test_other_functions()
test_problematic()
test_sanity_checks()
