"""
"""
from __future__ import division, print_function, absolute_import

import functools

import numpy as np
from numpy.testing import (run_module_suite, TestCase, assert_allclose,
        assert_almost_equal)

import accelem
from accelem._squarem import fixed_point_squarem as squarem


def test_scalar_trivial():
    # f(x) = 2x; fixed point should be x=0
    def func(x):
        return 2.0*x
    x0 = 1.0
    x = squarem(func, x0).x
    assert_almost_equal(x, 0.0)

def test_scalar_basic1():
    # f(x) = x**2; x0=1.05; fixed point should be x=1
    def func(x):
        return x**2
    x0 = 1.05
    x = squarem(func, x0).x
    assert_almost_equal(x, 1.0)

def test_scalar_basic2():
    # f(x) = x**0.5; x0=1.05; fixed point should be x=1
    def func(x):
        return x**0.5
    x0 = 1.05
    x = squarem(func, x0).x
    assert_almost_equal(x, 1.0)

def test_array_trivial():
    def func(x):
        return 2.0*x
    x0 = [0.3, 0.15]
    olderr = np.seterr(all='ignore')
    try:
        x = squarem(func, x0).x
    finally:
        np.seterr(**olderr)
    assert_almost_equal(x, [0.0, 0.0])

def test_array_basic1():
    # f(x) = c * x**2; fixed point should be x=1/c
    def func(x, c):
        return c * x**2
    c = np.array([0.75, 1.0, 1.25])
    x0 = [1.1, 1.15, 0.9]
    olderr = np.seterr(all='ignore')
    try:
        x = squarem(func, x0, args=(c,)).x
    finally:
        np.seterr(**olderr)
    assert_almost_equal(x, 1.0/c)

def test_array_basic2():
    # f(x) = c * x**0.5; fixed point should be x=c**2
    def func(x, c):
        return c * x**0.5
    c = np.array([0.75, 1.0, 1.25])
    x0 = [0.8, 1.1, 1.1]
    x = squarem(func, x0, args=(c,)).x
    assert_almost_equal(x, c**2)


if __name__ == "__main__":
    run_module_suite()

