import pytest
import numpy as np
from VayDiff import BasicMath as bm
from VayDiff.VayDiff import Diff
from VayDiff.VayDiff import Variable

def f1(x,y,z=2):
    return z+-z+x**2*y + z

def f2(x,y,z=2):
    return z+-z+5*x + bm.sin(y) + z

def f3(x,y,z=2):
    return x*y+y+z

def test_jacobian_22():
    x = Variable(val=3, name='x')
    y = Variable(val=5, name='y')
    t1 = Diff().jacobian([f1,f2], [x,y])
    assert(t1.shape == (2,2))
    assert(t1[0][0] == 30)
    assert(t1[0][1] == 9)
    assert(t1[1][0] == 5)
    assert(t1[1][1] == bm.cos(5))

def test_jacobian_23():
    x = Variable(val=3, name='x')
    y = Variable(val=5, name='y')
    z = Variable(val=2, name='z')
    t1 = Diff().jacobian([f1,f2], [x,y,z])
    assert(t1.shape == (2,3))
    assert(t1[0][0] == 30)
    assert(t1[0][1] == 9)
    assert(t1[0][2] == 1)
    assert(t1[1][0] == 5)
    assert(t1[1][1] == bm.cos(5))
    assert(t1[0][2] == 1)

def test_non_alphabetical_22():
    x = Variable(val=3, name='z')
    y = Variable(val=5, name='a')
    t1 = Diff().jacobian([f1,f2], [x,y])
    assert(t1.shape == (2,2))
    assert(t1[0][0] == 30)
    assert(t1[0][1] == 9)
    assert(t1[1][0] == 5)
    assert(t1[1][1] == bm.cos(5))

def test_jacobian_32():
    x = Variable(val=3, name='c')
    y = Variable(val=5, name='a')
    z = Variable(val=5, name='t')
    t1 = Diff().jacobian([f1,f2,f3], [x,y])
    assert(t1.shape == (3,2))
    assert(t1[0][0] == 30)
    assert(t1[0][1] == 9)
    assert(t1[1][0] == 5)
    assert(t1[1][1] == bm.cos(5))
    assert(t1[2][0] == 5)
    assert(t1[2][1] == 4)

test_jacobian_22()
test_jacobian_23()
test_non_alphabetical_22()
test_jacobian_32()
