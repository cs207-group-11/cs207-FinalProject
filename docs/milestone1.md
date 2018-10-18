# Milestone 1: Group 11

## Introduction

Differentiation is an increasingly important technique in scientific computing [1]. One of the important applications of differentiation is gradient descent, a technique that is extensively employed in machine learning. Here, the gradient (i.e. multivariate differentiation) of a given function is used to find the minimum of a given cost function [2]. 

Despite its emerging importance, traditional approaches to compute the derivatives involve slow and complicated methods such as symbolic differentiation and numerical differentiation.  Both these approaches perform poorly on high-order and multivariate derivatives, which are essential for gradient-based optimization problems. 

In this project, we introduce a software-based method to enable automatic differentiation to efficiently evaluate the derivative of a function. In addition to computing derivatives automatically, advanced methods such as back propagation will be included for complicated applications such as training a neural network. All of these methods will be incorporated in a well-documented Python package, `AutoDiff`, that can be easily installed and allows users to perform a variety of tasks such as Newton’s method and gradient descent.

## Background

## How to Use *AutoDiff*
First, the user needs to install Autodiff package via command line interface using one of the following commands:

`pip install AutoDiff`

or

`easy_install AutoDiff`

After installing this package, the user needs to import it into their project in order to fully utilize its functionality by running the following command:

```python
import AutoDiff as ad
```
In order to create a `AutoDiff` object, we need to instantiate it by calling the constructor as follows:

```python
result = ad.auto_diff(function, eval_point, order)
```

The function is a user-predefined function that needs to be differentiated, and `eval_point` is the point which the derivative will be computed at. The last argument is the order of derivative that the user wants to compute, and by default this value is set to 1. For multivariate differentiation,  `eval_point` will be a Python dictionary composed of key-value pairs. Each pair consists of variable name (e.g. ‘x’ or ‘y’), and its associated numerical value.

The `result` variable is always a Python list, and its first element is the nominal function value evaluated at `eval_poin`t. Other elements are different order of derivatives, with the second element being the first order derivative, the third element being the second order derivative and so on.

## Software Organization

We will organize the directory structure looks like follows: 

 AutoDiff\
         AutoDiff\
               __init__.py
               AutoDiff.py
               tests/
                    __init__.py
       test.py
         README.md
         setup.py 
         LICENSE
	 
In this directory, we have one Python module named AutoDiff.py. This file consists of all the algorithms and data structures and is the core of this project. In addition, we plan to include Numpy in our project to support scientific computation of elementary functions (which are outlined in the implementation section).

A series of tests will be written to provide full coverage of all the functions and classes defined in AutoDiff. They will be stored in the tests folder. In order to facilitate code integration, we will use `TravisCI` and `Coveralls` to automate the testing process for every commit and push to the Github repository.

Finally, this package will be distributed through PyPI. This enables the user to conveniently install the package using `pip `or `easy_install` command.


## Implementation

Our main class will be the `AutoDiff` class, however we will also use a `DualNumber` class under the hood to compute derivatives in the forward mode. This class will not be exposed to the user, however it is essential to the internal workings of the `AutoDiff` class. The most important function within the `AutoDiff` class is the `auto_diff` method: `auto_diff(function, eval_point, order)`. This function takes as input a `function`, an `eval_point` and an `order` (first derivative, second derivative and so on, defaults to 1). We imagine that a function (for a single variable) can be defined as follows:

```python
def my_function(x):
	return ad.power(x, 3) + 4
```

The function above basically represents f(x) = x^3 + 4. It is imperative that the function’s input matches the eval_point we are given. For instance, if `my_function` accepts a single numeric variable, `eval_point` in the auto_diff function cannot be a list of variables. One subtlety is our usage of `ad.power()` instead of `numpy.power()` in the function. Our aim is to write a wrapper for a few different basic elements of `numpy`, including the following: 

+ Basic functions: `add`, `subtract`, `multiply`, `divide`, `power`, `sqrt`
+ Trigonometric functions: `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`
+ Exponents and logarithms: `exp`, `log`, `log2`, `log10`

Each function would treat its inputs as dual numbers. This means that we need to know the value as well as the derivative of the function at the input point. We compute the value of the function by calling numpy’s corresponding function. For the elementary functions listed above, we are in the advantageous position of knowing the form of the first derivative. For instance, we know analytically that the derivative of sin(x) is cos(x). Within each of our wrapper functions, we store the analytical derivative and return a dual number consisting of the value of the function as well as the derivative, which is where our implementation of basic functions differs from that of numpy. In addition, we will overload basic operations such as `+,-,*, /` accordingly. 

At the moment, we do not consider arbitrary functions defined by the user including while loops, if statements and recursions. We require that any function be composed of basic functions we defined in the `auto_diff` class. However, an interesting extension of this project would be to compute gradients of arbitrary functions that do not include only basic functions, and a great example of a library that deals with these is the [autograd python library](https://github.com/HIPS/autograd).

Our implementation allows for a scalar function of scalars, a scalar function of vectors, a vector function of scalars and a vector function of vectors. This is done by allowing users to input, in the function parameter, a single function or an array of functions, and in the eval_point parameter, by allowing users to input a scalar value or an dictionary of key value pairs of variables and their corresponding values.

Let us imagine that we are trying to compute the derivative of the function in the homework problem: `f = alpha*x+c`. We would write this function, after breaking it down using elementary elementary multiplication and addition functions within our AutoDiff class (with appropriate overloading).  as follows: 

```python
def f(x, alpha=2):
	return alpha*x+c
```

We would then compute the first derivative of this function as follows: 

```python
ad = AutoDiff()
x = 3 # or any other value of the user’s choice)
ad.auto_diff(function=f, eval_point=x, order=1)
```

This function should return the dual component of the dual number we store under the hood, which would be the chosen value of alpha.

## References
[1] M. T. Heath, “Scientific Computing: An Introductory Survey Chapter 8 - Numerical Integration and Differentiation,”[Online]. Accessed October 18th, 2018. Available: http://heath.cs.illinois.edu/scicomp/notes/chap08.pdf
[2] S. Ruder (2017, Jun, 15th), “An Overview of Gradient Descent Optimization Algorithms,”  *arXiv*.[Online]. Access October 18th, 2018. Available: https://arxiv.org/pdf/1609.04747.pdf

