"""
accelem module short description first line

"""
from __future__ import division, print_function, absolute_import

from ._squarem import *
from .linesearch import *

__all__ = [s for s in dir() if not s.startswith('_')]

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench

__all__ += ['test', 'bench']
