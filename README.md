[![Build Status](https://travis-ci.com/cs207-group-11/cs207-FinalProject.svg?branch=master)](https://travis-ci.com/cs207-group-11/cs207-FinalProject)[![Coverage Status](https://coveralls.io/repos/github/cs207-group-11/cs207-FinalProject/badge.svg?branch=master)](https://coveralls.io/github/cs207-group-11/cs207-FinalProject?branch=master)

# VayDiff

An Automatic Differentiation Library for Python 3. This project was done for CS207 at Harvard University, taught by Professor David Sondak.

## How to Install *VayDiff*

### Installing via PyPI (for end-users)

Download our project on [PyPI](https://pypi.org/project/VayDiff/) using the following command:

```
pip install VayDiff
```

### Manual Installation (for developers)

Clone or download our [GitHub repository](https://github.com/HIPS/autograd) and navigate into this directory in your terminal.

Optional: create a virtual environment using `virtualenv`. This can be downloaded using `pip3` or `easy_install` as follows:

```
pip3 install virtualenv
```

or

```
sudo easy_install virtualenv
```

Then, create a virtual environment (using Python3), activate this virtual environment, and install the dependencies as follows:

```
virtualenv -p python3 my_env
source my_env/bin/activate
pip3 install -r requirements.txt
```

In order to deactivate the virtual environment, use the following command

```
deactivate
```

## A small example

```python
from VayDiff.VayDiff import Variable
from VayDiff.VayDiff import Diff

def user_function(a):
  return a**2

x = Variable(3, name='x')
t = Diff().auto_diff(user_function, [x])
print(t.val, t.der['x'])
9 6.0
```

## A more interesting example: Fractals!

We provide a Graphical User Interface for our feature.
![GUI](./docs/Final/Figures/Interface.png)

When we enter the function f(x)=x<sup>6</sup>-1, we will get the picture below:
![newton_fractal](./docs/Final/Figures/fractal.gif)

When we click the `show gradient` button, we will get the animation of the root finding process. 
![animation](./docs/Final/Figures/animation_cleaned.gif)


## Made By:

1. Abhimanyu Vasishth
2. Zheyu Wu
3. Yiming Xu
