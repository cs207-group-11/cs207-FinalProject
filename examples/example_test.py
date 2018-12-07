import numpy as np
from VayDiff import BasicMath as bm
from VayDiff import VayDiff as ad

if __name__ == '__main__':
    x = ad.Variable(3, name='x')
    t = x + 2
    print(t.val, t.der['x'])
    assert(t.val == 5)
    assert(t.der['x'] == 1.0)
