import pytest
import numpy as np
from VayDiff import BasicMath as bm
from VayDiff.VayDiff import Diff
from VayDiff.VayDiff import Variable

def simple_op_complete (x,y):
    return 4 + x*x + x**2 + (-x) + y*y + 1**y + 3

def log_exp(x,y):
    return bm.log(x*x) + x*bm.exp(y)

def logk_test(x):
    return bm.logk(x*x, base=10)

def sqrt_fn(x,y):
    return x*bm.sqrt(x*y) + x*x*x

def sin_cos(x,y):
    return y*bm.sin(x*x*x) + x*bm.cos(y)

def tan_fn(x,y):
    return y*bm.tan(x*y)

def rpow_fn(x,y):
    return x**y

def test_operators():
    a = Variable(val=3, name='a')
    b = Variable(val=2, name='b')

    t1 = Diff().auto_diff(function = simple_op_complete, eval_point = [a,b])
    assert(t1.sec_der['a'] == 4)
    assert(t1.sec_der['b'] == 2)

    t2 = Diff().auto_diff(function = log_exp, eval_point = [a,b])
    np.testing.assert_approx_equal(t2.sec_der['a'], -2/9)
    assert(t2.sec_der['b'] == 3*np.exp(2))

    t3 = Diff().auto_diff(function = sqrt_fn, eval_point = [a,b])
    np.testing.assert_approx_equal(t3.sec_der['a'], 18+6/(4*np.sqrt(6)))
    np.testing.assert_approx_equal(t3.sec_der['b'], -27/(4*(6**1.5)))

    t4 = Diff().auto_diff(function = sin_cos, eval_point = [a,b])
    np.testing.assert_approx_equal(t4.sec_der['a'], -1404.91310072818349654572310)
    np.testing.assert_approx_equal(t4.sec_der['b'], -3*np.cos(2))

    t5 = Diff().auto_diff(function = tan_fn, eval_point = [a,b])
    np.testing.assert_approx_equal(t5.sec_der['a'], -5.0503989647387223596)
    np.testing.assert_approx_equal(t5.sec_der['b'], -4.85529005011658178)

    t6 = Diff().auto_diff(function = rpow_fn, eval_point = [a,b])
    np.testing.assert_approx_equal(t6.sec_der['a'], 2)
    np.testing.assert_approx_equal(t6.sec_der['b'], 9*np.log(3)**2)

    t7 = Diff().auto_diff(function = logk_test, eval_point = [a])
    np.testing.assert_approx_equal(t7.sec_der['a'], -2/(9*np.log(10)))

def arcsin_fn(x,y):
    return -y*bm.arcsin(x*y)

def arccos_fn(x,y):
    return -y*bm.arccos(x*y)

def arctan_fn(x,y):
    return -y*bm.arctan(x*y)

def test_invtrig():
    a = Variable(val=1, name='a')
    b = Variable(val=0.5, name='b')

    t1 = Diff().auto_diff(function = arcsin_fn, eval_point = [a,b])
    np.testing.assert_approx_equal(t1.sec_der['a'], -0.096225, significant=5)
    np.testing.assert_approx_equal(t1.sec_der['b'], -2.6943, significant=5)

    t2 = Diff().auto_diff(function = arccos_fn, eval_point = [a,b])
    np.testing.assert_approx_equal(t2.sec_der['a'], 0.096225, significant=5)
    np.testing.assert_approx_equal(t2.sec_der['b'], 2.6943, significant=5)

    t3 = Diff().auto_diff(function = arctan_fn, eval_point = [a,b])
    np.testing.assert_approx_equal(t3.sec_der['a'], 0.08, significant=2)
    np.testing.assert_approx_equal(t3.sec_der['b'], -1.28, significant=3)

def sinh_fn(x,y):
    return -x*y*bm.sinh(x*y)

def cosh_fn(x,y):
    return -x*y*bm.cosh(x*y)

def tanh_fn(x,y):
    return -x*y*bm.tanh(x*y)

def test_hyp():
    a = Variable(val=1, name='a')
    b = Variable(val=0.5, name='b')

    t1 = Diff().auto_diff(function = sinh_fn, eval_point = [a,b])
    np.testing.assert_approx_equal(t1.sec_der['a'], -0.62895, significant=5)
    np.testing.assert_approx_equal(t1.sec_der['b'], -2.5158, significant=5)

    t2 = Diff().auto_diff(function = cosh_fn, eval_point = [a,b])
    np.testing.assert_approx_equal(t2.sec_der['a'], -0.401501, significant=6)
    np.testing.assert_approx_equal(t2.sec_der['b'], -1.606, significant=4)

    t3 = Diff().auto_diff(function = tanh_fn, eval_point = [a,b])
    np.testing.assert_approx_equal(t3.sec_der['a'], -0.302366, significant=6)
    np.testing.assert_approx_equal(t3.sec_der['b'], -1.20946, significant=6)

test_operators()
test_invtrig()
test_hyp()
