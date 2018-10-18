# Milestone 1: Group 11

## Introduction 

## Background

## How to Use *AutoDiff*

## Software Organization

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


