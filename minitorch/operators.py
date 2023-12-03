"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    r""" "
    Description: Multiplication
    Multiplies the values x and y as such f(x,y) = x * y = z

        Args:
            x: float
            y: float
        Returns:
            The sum of x and y
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    return x * y


def id(x: float) -> float:
    r""" "
    Description: Indentity
    f(x) = x

        Args:
            x: float

        Returns:
            The same value of x
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    return x


def add(x: float, y: float) -> float:
    r""" "
    Description: Addition
    Adds two values together f(x,y) = x + y = z

        Args:
            x: float
            y: float
        Returns:
            The sum of x and y
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    return x + y


def neg(x: float) -> float:
    r""" "
    Description: Negative Function
    Makes the value negative as such f(x) = -x

        Args:
            x: float

        Returns:
            The negative of the given value
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    return -x


def lt(x: float, y: float) -> float:
    r""" "
    Description: Less Then Function
    Compares the values of x and y like : f(x) =$ 1.0 if x is less than y else 0.0

        Args:
            x: float
            y: float
        Returns:
            Either 1.0 if x is less than y and 0.0 if x is greater than y.
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    if x < y:
        return 1.0
    else:
        return 0.0


def eq(x: float, y: float) -> float:
    r""" "
    Description: Equal to Function
    Looks to see if the values of x and y are the same: f(x) =$ 1.0 if x is equal to y else 0.0

        Args:
            x: float
            y: float
        Returns:
            Either 1.0 if x is the same as y and 0.0 if not.
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    if x == y:
        return 1.0
    else:
        return 0.0


def max(x: float, y: float) -> float:
    r""" "
    Description: Greater Than Function
    Compares x and y to see if x is greater than y like: f(x) =$ x if x is greater than y else y.

        Args:
            x: float
            y: float
        Returns:
            Either 1.0 if x is greater than y and 0.0 if not.
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    if x > y:
        return x
    else:
        return y


def is_close(x: float, y: float) -> float:
    r""" "
    Description: Is Close
    Checks to see if x and y are almost equal: f(x,y) = |x-y| <= 0.0

        Args:
            x: float
            y: float
        Returns:
            The absolute value of x and y
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    r"""
    Description: Sigmoid
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
        Args:
            x: float
        Returns:
            The outcoem of x as it is applied to the Sigmoid Function.

    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    r"""
    DEscription: f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)

    Args:
            x: float
        Returns:
            The outcome of x as it is applied to ReLu
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    if x > 0.0:
        return x
    else:
        return 0.0


EPS = 1e-6


def log(x: float) -> float:
    r"""
    Description: Logarithmic Function
        Takes the base 10 log x: f(x) = log(x)

        Args:
            x: float
        Returns:
            The outcoem of x as it is applied to the logarithmic function.
    """
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    r"""
    Description: Exponent
        Creates a function that is the exponent: $f(x) = e^{x}

        Args:
            x: float
        Returns:
            The outcome of e raised to the x.
    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"""
    Description: Logarithmic Back Function
    Takes output of the log function and then multiples a floating point by the
    derivative of f = log$ as above, compute $d \times f'(x)

        Args:
            x: float
            d: float
        Returns:
            Reuren the value of d divided by the value of x
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")

    return d / x


def inv(x: float) -> float:
    r"""
    Description: Inverse Function
    Takes a value x and inverts the value making it: f(x) = 1/x

        Args:
            x: float
        Returns:
            The value of 1/x
    """

    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    r"""
    Description: Inverse Back Function
    Takes the outpout of the inverse function and multiplies d by it derivative: f(x) = 1/x$ compute $d times f'(x)

        Args:
            x: float
            d: float
        Returns:
            The product of d times the derivative of the inverse function
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    return -(d) / (x**2)


def relu_back(x: float, d: float) -> float:
    r"""
    Description: ReLu Back Function
    Takes the output of the ReLu function and multiplies d by it derivative:nd
        Args:
            x: float
            d: float
        Returns:
            The product of d times the derivative of the ReLu function
    """

    # TODO: Implement for Task 0.1.
    # raise NotImplementedError("Need to implement for Task 0.1")
    if x < 0.0:
        return d * 0.0
    elif x >= 0.0:
        return d * 1.0
    else:
        return 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    r"""
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError("Need to implement for Task 0.3")
    def action(ls: Iterable[float]) -> Iterable[float]:
        nlist = []

        for i in ls:
            nlist.append(fn(i))
        return nlist

    return action


def negList(ls: Iterable[float]) -> Iterable[float]:
    r"""
    Description: Negative List Function
    Uses the funcitons map and neg to negate each element in the list
        Args:
            fn:
        Returns:
            Either 1.0 or 0.0 depending on the values of x and y
    """
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError("Need to implement for Task 0.3")

    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    r"""
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError("Need to implement for Task 0.3")

    def action(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        r"""
        Description: action
            Zips the two given lists into a new list and which it then applies the function to
        Args:
            ls1: list
            ls2: list
        Returns:
            The new list applying the function that new list
        """
        nlist = []
        for i, j in zip(ls1, ls2):
            nlist.append(fn(i, j))
        return nlist

    return action


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    r"""
    Description: AddLists
    Add the elements of `ls1` and `ls2` using `zipWith` and `add`

        Args:
            ls1: list
            ls2: list
        Returns:
            Returns the sum of the two lists
    """
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError("Need to implement for Task 0.3")

    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError("Need to implement for Task 0.3")

    def action(ls: Iterable[float]) -> float:
        r"""
        Description: action
        applys a function to each element of the list

        Args:
           ls: list
        Returns:
           returns the list with the function applied to the start value.
        """
        val = start
        for i in ls:
            val = fn(val, i)
        return val

    return action


def sum(ls: Iterable[float]) -> float:
    r"""
    Description: Sum
    Sum up a list using `reduce` and `add`

        Args:
            fn: add
            start: float
            ls: list
        Returns:
            The product of reduce and mul with start value 1.0
    """
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError("Need to implement for Task 0.3")
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    r"""
    Description: Product
    Takes the product of a list using `reduce` and `mul`

        Args:
            fn: multiply
            start: float
            ls: list
        Returns:
            The product of reduce and mul with start value 1.0
    """
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError("Need to implement for Task 0.3")
    return reduce(mul, 1.0)(ls)