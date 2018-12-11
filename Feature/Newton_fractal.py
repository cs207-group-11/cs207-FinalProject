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
    '''This function returns the RGB color value given the index of the root
    and the number of iterations needed to arrive the root'''

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
    '''This function implements the Newton's method given the input functiono as well as the
    starting point. Thresholds such as the max iteration count and the eps also need to be
    specified'''
    ad = AD.Diff()


    for i in range(max_iter):
        x = AD.Variable(c, name='x')
        c2 = c - coef * f(c) / (ad.auto_diff(function = f, eval_point = [x])).der['x']

        if abs(c2 - c) < eps:
            return c2, i
        c = c2

    return None, None

def draw(f, size, name, x_min=-2.0, x_max=2., y_min=-2.0, y_max=2.0, eps=1e-6, max_iter=40):
    '''This function generates the image of Newton's Fractal given the file name, file Size and the
    function of interest.'''
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
    '''This function dynamically plots the convergence path when Newton's method is used. Similar to the
    previous function, the user needs to specify the function, file size and file name.'''
    img = mpimg.imread(name)

    plt.ion()
    plt.imshow(img, extent=[-img.shape[1]/2., img.shape[1]/2., -img.shape[0]/2., img.shape[0]/2. ])

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
    '''This function implements and launches the Graphical User Interface powered by Tkinter'''
    root = Tk()
    root.title("Newton's Fractal Generator")


    #Labels
    label_1 = Label(root, text = "Input Equation")
    label_2 = Label(root, text = "Image Size")
    label_3 = Label(root, text = "Image Name")

    #Entries
    entry_1 = Entry(root)
    entry_2 = Entry(root)
    entry_2.insert(END, '500')
    entry_3 = Entry(root)
    entry_3.insert(END, 'output.png')

    #Button
    button_1 = Button(root, text = "Run", command = lambda: run_command(entry_1, entry_2, entry_3))
    button_2 = Button(root, text = "Show Image", command = lambda: image_command(entry_3))
    button_3 = Button(root, text = "Show Gradient", command = lambda: gradient_command(entry_1, entry_2, entry_3))

    #Placements
    label_1.grid(row = 1)
    label_2.grid(row = 2)
    label_3.grid(row = 3)


    entry_1.grid(row = 1, column = 1)
    entry_2.grid(row = 2, column = 1)
    entry_3.grid(row = 3, column = 1)
    button_1.grid(row = 4, column = 0)
    button_2.grid(row = 4, column = 1)
    button_3.grid(row = 4, column = 2)

    root.mainloop()

def run_command(entry_1, entry_2, entry_3):
    '''This is the function that will be called once the "Run" button is clicked. This function is
    a wrapper function for the draw function'''
    equation = entry_1.get()
    img_size = int(entry_2.get())

    img_name = entry_3.get()

    g = lambda x: eval(equation)
    draw(g, img_size, img_name)

def image_command(entry_3):
    '''This function is called when the "Show" button is clicked. This function simply shows the image
    generated by the Newton's Fractal'''
    img_name = entry_3.get()
    img = mpimg.imread(img_name)
    plt.ion()
    plt.imshow(img, extent=[-img.shape[1]/2., img.shape[1]/2., -img.shape[0]/2., img.shape[0]/2. ])


def gradient_command(entry_1, entry_2, entry_3):
    '''This function is called when "Show Gradient" button is pressed. It is a wrapper function
    for the show_route() function'''
    equation = entry_1.get()
    img_size = int(entry_2.get())
    img_name = entry_3.get()

    g = lambda x: eval(equation)
    show_route(g, img_size, img_name)

if __name__ == '__main__':
    launcn_UI()
