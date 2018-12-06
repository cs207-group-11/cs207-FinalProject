import pytest
import numpy as np
from AutoDiff import BasicMath as bm
from AutoDiff.AutoDiff import Diff
from AutoDiff.AutoDiff import Variable

epsilon = 1e-7

# Simple operations
def add_function(x,y):
    return 4+x+y+2

def sub_function(x,y):
    return x-y-1

def mul_function1(x,y):
    return 2*x*y*3

def mul_function2(x,y):
    return 7.0 + 3*x - y*2 - 4.0

def pow_function1(x,y):
    return x**2 + y**2

def rpow_function(x,y):
    return 2**x + 3**y

def div_scalar(x, y):
    return x/2 + y/4

def div_function(x,y):
    return x/2 + 6/y - 1

def rsub_function(x,y):
    return 1-x-y

def neg_pow_function(x,y):
    return x**-1 + y**-1

def pow_function2(x,y):
    return x**y

def simple_op_complete(x,y):
    return x*x + x/x + 3*x*2 - 3*(x+2) + y/y - y*y

def unary(x,y):
    return 3*(+x) + (-y)

# Testing addition, subtraction, multiplication, division and power
def test_simple_operators():
    x = Variable(val=2, name='x')
    y = Variable(val=1, name='y')
    t1 = Diff().auto_diff(function = add_function, eval_point = [x,y])
    assert(t1.val == 9 and t1.der['x'] == 1 and t1.der['y'] == 1)

    t2 = Diff().auto_diff(function = sub_function, eval_point = [x,y])
    assert(t2.val == 0 and t2.der['x'] == 1 and t2.der['y'] == -1)

    t3 = Diff().auto_diff(function = mul_function1, eval_point = [x,y])
    assert(t3.val == 12 and t3.der['x'] == 6 and t3.der['y'] == 12)

    t4 = Diff().auto_diff(function = mul_function2, eval_point = [x,y])
    assert(t4.val == 7 and t4.der['x'] == 3 and t4.der['y'] == -2)

    t5 = Diff().auto_diff(function = pow_function1, eval_point = [x,y])
    assert(t5.val == 5 and t5.der['x'] == 4 and t5.der['y'] == 2)

    t6 = Diff().auto_diff(function = rpow_function, eval_point = [x,y])
    assert(t6.val == 7 and t6.der['x'] == 4*np.log(2) and t6.der['y'] == 3*np.log(3))

    t7 = Diff().auto_diff(function = div_scalar, eval_point = [x,y])
    assert(t7.val == 1.25 and t7.der['x'] == 0.5 and t7.der['y'] == 0.25)

    t8 = Diff().auto_diff(function = div_function, eval_point = [x,y])
    assert(t8.val == 6 and t8.der['x'] == 0.5 and t8.der['y'] == -6)

    t9 = Diff().auto_diff(function = rsub_function, eval_point = [x,y])
    assert(t9.val == -2 and t9.der['x'] == -1 and t9.der['y'] == -1)

    t10 = Diff().auto_diff(function = neg_pow_function, eval_point = [x,y])
    assert(t10.val == 1.5 and t10.der['x'] == -0.25 and t10.der['y'] == -1)

    t11 = Diff().auto_diff(function = pow_function2, eval_point = [x,y])
    assert(t11.val == 2 and t11.der['x'] == 1 and t11.der['y'] == 2*np.log(2))

    t12 = Diff().auto_diff(function = simple_op_complete, eval_point = [x,y])
    assert(t12.val == 5 and t12.der['x'] == 7 and t12.der['y'] == -2)

    t13 = Diff().auto_diff(function = unary, eval_point = [x,y])
    assert(t13.val == 5 and t13.der['x'] == 3 and t13.der['y'] == -1)

# Other functions
def log_exp_function(x,y):
    return bm.log(x) + bm.exp(y)

def sqrt_function(x,y):
    return bm.sqrt(x) + bm.sqrt(y)

def test_bm_vector():
    a = Variable(val=2, name='x')
    b = Variable(val=3, name='y')

    t1 = Diff().auto_diff(function = log_exp_function, eval_point = [a,b])
    assert(t1.val == np.log(2) + np.exp(3))
    assert(t1.der['x'] == 0.5)
    assert(t1.der['y'] == np.exp(3))

    t2 = Diff().auto_diff(function = sqrt_function, eval_point = [a,b])
    np.testing.assert_approx_equal(t2.val, np.sqrt(2) + np.sqrt(3))
    np.testing.assert_approx_equal(t2.der['x'],1/(2*np.sqrt(2)))
    np.testing.assert_approx_equal(t2.der['y'],1/(2*np.sqrt(3)))

# Trig Functions
def sin_cos(x):
    return bm.sin(x) + bm.cos(x)

def tan_function(x):
    return bm.tan(2*x)

def inverse_trig(x):
    return bm.arcsin(x) - bm.arccos(x) + bm.arctan(x)

def test_trig_vector():
    a = Variable(val=0, name='x')
    t1 = Diff().auto_diff(function = sin_cos, eval_point = [a])
    assert(t1.val == 1)
    assert(t1.der['x'] == 1)

    t2 = Diff().auto_diff(function = tan_function, eval_point = [a])
    assert(t2.val == 0)
    assert(t2.der['x'] == 2)

    t3 = Diff().auto_diff(function = inverse_trig, eval_point = [a])
    np.testing.assert_approx_equal(t3.val, -np.pi/2)
    np.testing.assert_approx_equal(t3.der['x'], 3)

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

test_simple_operators()
test_bm_vector()
test_sanity_checks()
test_trig_vector()
