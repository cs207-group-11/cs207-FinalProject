# CREDITS: This code is taken from Professor David Sondak's Lecture 9 code
import numpy as np
from VayDiff import BasicMath as bm
from VayDiff import VayDiff as ad

def f(x):
    # Hard-coded f(x)
    return x - bm.exp(-2.0 * bm.sin(4.0*x) * bm.sin(4.0*x))

# Start Newton algorithm
xk = 0.1 # Initial guess
tol = 1.0e-08 # Some tolerance
max_it = 100 # Just stop if a root isn't found after 100 iterations

root = None # Initialize root
for k in range(max_it):
    t = ad.Diff().auto_diff(function = f, eval_point = [ad.Variable(xk, name='xk')])
    delta_xk = -f(xk) / t.der['xk'] # Update Delta x_{k}
    if (abs(delta_xk) <= tol): # Stop iteration if solution found
        root = xk + delta_xk
        print("Found root at x = {0:17.16f} after {1} iteratons.".format(root, k+1))
        break
    print("At iteration {0}, Delta x = {1:17.16f}".format(k+1, delta_xk))
    xk += delta_xk # Update xk
