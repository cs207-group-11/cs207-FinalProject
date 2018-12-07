import os
import sys
from PIL import Image
from tqdm import tqdm

module_path = os.path.join(os.path.abspath('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from AutoDiff import AutoDiff as AD


def color(ind, level):

    colors = [(180, 0, 30), (0, 180, 30), (0, 30, 180),
              (0, 190, 180), (180, 0, 175), (180, 255, 0),
              (155, 170, 180), (70, 50, 0),
              (150, 60, 0), (0, 150, 60), (0, 60, 150),
              (60, 150, 0), (60, 0, 150), (150, 0, 60),
              (130, 80, 0), (80, 130, 0), (130, 0, 80),
              (80, 0, 130), (0, 130, 80), (0, 80, 130),
              (110, 100, 0), (100, 110, 0), (0, 110, 100),
              (0, 100, 100), (110, 0, 100), (100, 0, 110),
              (255, 255, 255)]
    if ind < len(colors):
        c = colors[ind]
    else:
        c = (ind % 4 * 4, ind % 8 * 8, ind % 16 * 16)
    if max(c) < 210:
        c0 = c[0] + level
        c1 = c[1] + level
        c2 = c[2] + level
        return (c0, c1, c2)
    else:
        return c

def newton_method(f, c, max_iter, eps):
    # c is the initial value using Newton's method
    # max_iter is the iteration threshold
    # eps is the stopping threshold

    # initialize AutoDiff object
    ad = AD.Diff()

    for i in range(max_iter):
        x = AD.Variable(c, name='x')
        c2 = c - f(c) / (ad.auto_diff(function = f, eval_point = [x])).der['x']

        if abs(c2 - c) < eps:
            return c2, i
        c = c2

    return None, None

def draw(f, size, name, x_min=-2.0, x_max=2., y_min=-2.0, y_max=2.0, eps=1e-6,
         max_iter=40):

    roots = []
    img = Image.new("RGB", (size, size))
    print ('Solving for roots using Newton\'s method')
    for x in tqdm(range(size)):
        for y in range(size):
            z_x = x_min + x * (x_max - x_min) / (size - 1)
            z_y = y_min + y * (y_max - y_min) / (size - 1)

            root, n_converge = newton_method(f, complex(z_x, z_y), max_iter, eps)
            if root:
                cached_root = False
                for r in roots:
                    if abs(r - root) < 1e-4:
                        root = r
                        cached_root = True
                        break
                if not cached_root:
                    roots.append(root)
            if root:
                img.putpixel((x, y), color(roots.index(root), n_converge)) # 上色
    print(roots)
    img.save(name, "PNG")

def f(x):
    return x ** 8 + 15 * x ** 4 - 16

if __name__ == '__main__':
    draw(f, 2000, "output.png")
