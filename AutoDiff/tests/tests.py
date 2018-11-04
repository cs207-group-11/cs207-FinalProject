import pytest
from AutoDiff.AutoDiff import ad

def f1(x):
    return x**2

def f2(x):
    return 10+x*3+4

def test_init():
    x = 3
    t1 = ad().auto_diff(function = f1, eval_point = x)
    assert(t1.val == 9)
    assert(t1.der == 6)

    t2 = ad().auto_diff(function = f2, eval_point = x)
    assert(t2.val == 23)
    assert(t2.der == 3)

test_init()
