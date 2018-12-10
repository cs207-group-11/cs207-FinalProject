import os
import sys
from tkinter import *
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

module_path = os.path.join(os.path.abspath('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from VayDiff import VayDiff as AD
from VayDiff.BasicMath import *

def color(ind, level):

    colors = [(36,136,136), (231,71,94), (240,216,121),
              (230,230,230), (255,192,190), (184,208,10),
              (88,116,152),(255,200,87),
              (155, 170, 180), (70, 50, 0),
              (150, 60, 0), (0, 150, 60), (0, 60, 150),
              (60, 150, 0), (60, 0, 150), (150, 0, 60),
              (130, 80, 0), (80, 130, 0), (130, 0, 80),
              (80, 0, 130), (0, 130, 80), (0, 80, 130),
              (110, 100, 0), (100, 110, 0), (0, 110, 100),
              (0, 100, 100), (110, 0, 100), (100, 0, 110),
              (255, 255, 255)]

    if ind < len(colors):
        base_color = colors[ind]
        const_adjuster0 = (-4, -4, -4)
        const_adjuster1 = (-8, -8, -8)
        const_adjuster2 = (-16, -16, -16)
        const_adjuster3 = (-24, -24, -24)
        if level < 1:
            c = base_color
        elif level < 2:
            c = tuple(map(sum, zip(base_color, const_adjuster0)))
        elif level < 5:
            c = tuple(map(sum, zip(base_color, const_adjuster1)))
        elif level < 8:
            c = tuple(map(sum, zip(base_color, const_adjuster2)))
        else:
            c = tuple(map(sum, zip(base_color, const_adjuster3)))

    else:
        c = (ind % 4 * 4, ind % 8 * 8, ind % 16 * 16)
    if max(c) < 210:
        c0 = c[0] + level
        c1 = c[1] + level
        c2 = c[2] + level
        return (c0, c1, c2)
    else:
        return c

def newton_method(f, c, coef, max_iter, eps):
    # c is the initial value using Newton's method
    # max_iter is the iteration threshold
    # eps is the stopping threshold

    # initialize AutoDiff object
    ad = AD.Diff()


    for i in range(max_iter):
        x = AD.Variable(c, name='x')
        c2 = c - coef * f(c) / (ad.auto_diff(function = f, eval_point = [x])).der['x']

        if abs(c2 - c) < eps:
            return c2, i
        c = c2

    return None, None

def draw(f, size, name, x_min=-2.0, x_max=2., y_min=-2.0, y_max=2.0, eps=1e-6, max_iter=40):

    roots = []
    img = Image.new("RGB", (size, size))
    print ('Solving for roots using Newton\'s method')

    for x in tqdm(range(size)):
        for y in range(size):
            z_x = x_min + x * (x_max - x_min) / (size - 1)
            z_y = y_min + y * (y_max - y_min) / (size - 1)

            root, n_converge = newton_method(f, complex(z_x, z_y), 1, max_iter, eps)
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

def show_route(f, size, name, x_min=-2.0, x_max=2., y_min=-2.0, y_max=2.0, eps=1e-6, max_iter=40):
    img = mpimg.imread('old.png')

    plt.ion()
    plt.imshow(img, extent=[-img.shape[1]/2., img.shape[1]/2., -img.shape[0]/2., img.shape[0]/2. ])
    print (img)
    plt.xlim(-size/2,size/2)
    plt.ylim(-size/2,size/2)
    ax = plt.axes()
    for x in tqdm(range(size)):
        for y in tqdm(range(size)):
            z_x = x_min + x * (x_max - x_min) / (size - 1)
            z_y = y_min + y * (y_max - y_min) / (size - 1)

            c = complex(z_x, z_y)
            # Starts Newton's method
            ad = AD.Diff()
            for i in range(max_iter):
                temp = AD.Variable(c, name='x')
                c2 = c - f(c) / (ad.auto_diff(function = f, eval_point = [temp])).der['x']

            # Of course we will not draw all arrows for all points
                if (x % (size//10) == 0 and y % (size//10) == 0):
                    if (abs(c2.real-c.real) > 0.1 or abs(c2.imag - c.imag) > 0.1):
                        ax.arrow(c.real * size / 4, c.imag * size / 4, (c2.real-c.real) * size / 4, (c2.imag - c.imag) * size / 4, color = (177/255, 196/255, 226/255, 0.3), head_width= size // 100, head_length = size // 100)
                        plt.pause(0.01)
                if abs(c2 - c) < eps:
                    break
                c = c2

def launcn_UI():

    root = Tk()
    root.title("Newton's Fractal Generator")

    derivative_order = StringVar(root)
    derivative_order.set("1") # default value
    drop_down_menu = OptionMenu(root, derivative_order, "1", "2")

    #Labels
    label_1 = Label(root, text = "Input Equation")
    label_2 = Label(root, text = "Evaluation Point")
    label_3 = Label(root, text = "Image Size")
    label_4 = Label(root, text = "Derivative Order")

    #Entries
    entry_1 = Entry(root)
    entry_2 = Entry(root)
    entry_3 = Entry(root)
    entry_3.insert(END, '500')


    #Button
    button_1 = Button(root, text = "Run", command = lambda: run_command(entry_1, entry_2, entry_3, derivative_order))
    button_2 = Button(root, text = "Show Gradient", command = lambda: gradient_command(entry_1, entry_2, entry_3, derivative_order))

    #Placements
    label_1.grid(row = 1)
    label_2.grid(row = 2)
    label_3.grid(row = 3)
    label_4.grid(row = 4)


    entry_1.grid(row = 1, column = 1)
    entry_2.grid(row = 2, column = 1)
    entry_3.grid(row = 3, column = 1)
    drop_down_menu.grid(row = 4, column = 1)
    button_1.grid(row = 5, column = 0)
    button_2.grid(row = 5, column = 1)


    root.mainloop()

def run_command(entry_1, entry_2, entry_3, derivative_order):
    equation = entry_1.get()
    eval_point = float(entry_2.get())
    img_size = int(entry_3.get())
    deri_order = int(derivative_order.get())

    g = lambda x: eval(equation)
    draw(g, img_size, "old.png")


def run_command(entry_1, entry_2, entry_3, derivative_order):
    equation = entry_1.get()
    eval_point = float(entry_2.get())
    img_size = int(entry_3.get())
    deri_order = int(derivative_order.get())

    g = lambda x: eval(equation)
    show_route(g, img_size, "old.png")

def f(x):
    return x**6-1

if __name__ == '__main__':
    launcn_UI()
