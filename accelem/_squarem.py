"""
An accelerated fixed-point solver designed for smooth contraction mappings.

The most straightforward application of this solver is to accelerate
the convergence of expectation-maximization algorithms for maximum likelihood
estimation of parameters of statistical models.

"""
from __future__ import division, print_function, absolute_import

import functools

import numpy as np

from scipy.optimize import OptimizeResult


class _ConvergenceError(Exception):
    pass


class _counted_calls(object):
    """
    Tracks the number of calls.

    Raises an error if the max number of calls is exceeded.

    """
    def __init__(self, f, maxfun=None):
        self.f = f
        self.ncalls = 0
        self.maxfun = maxfun
    def __call__(self, *args, **kwargs):
        self.ncalls += 1
        if self.maxfun is not None and self.ncalls > self.maxfun:
            msg = 'exceeded %d function calls' % self.maxfun
            raise _ConvergenceError(msg)
        return self.f(*args, **kwargs)


def _step(x, r, v, alpha):
    return x - 2*alpha*r + alpha*alpha*v


def _modify_step_length(alpha, L, backtrack_rate, step):
    """
    Note that when alpha = -1 this corresponds to a pure x=func(x) step.
    Because of the Lyapunov theory, step lengths between 0 and -1 are stable.
    The step length modfication brings more extreme step lengths towards -1.

    """
    if alpha >= 0:
        raise _ConvergenceError('non-negative step length %f' % alpha)
    if alpha > -1:
        return alpha
    L0 = L(step(0))
    Ln = L(step(alpha))
    while L0 > Ln:
        alpha = (1-backtrack_rate)*alpha + backtrack_rate*(-1)
        Ln = L(step(alpha))
    return alpha


def fixed_point_squarem(func, x0, args=(), L=None, backtrack_rate=0.1,
        atol=1e-8, maxiter=500, maxfun=None):
    """
    Not globalized.

    Parameter list inspired by scipy.optimize.fixed_point.

    Parameters
    ----------
    func : function
        Function to evaluate.
    x0 : scalar or array_like
        Initial guess of fixed point of function.
    args : tuple
        Extra arguments to `func`.
    L : function, optional
        Underlying Lyapunov function to minimize.
    backtrack_rate : float, optional
        A tuning parameter between 0 and 1.
        This is not used if L is not provided.
    atol : float, optional
        Tolerance of norm of difference between values of successive iterations.
    maxiter : int, optional
        Maximum number of iterations.
    maxfun : int, optional
        Maximum number of function calls.

    Returns
    -------
    result : OptimizeResult object
        Highlights include result.x, result.success, and result.message.

    """
    if not (0 <= backtrack_rate <= 1):
        raise ValueError('backtrack rate should be between 0 and 1')
    if np.isscalar(x0):
        norm = np.absolute
    if not np.isscalar(x0):
        x0 = np.asarray(x0)
        norm = np.linalg.norm
    func = _counted_calls(func, maxfun)
    xn = x0
    try:
        completed_iterations = 0
        while True:
            x1 = func(x0, *args)
            x2 = func(x1, *args)
            r = x1 - x0
            v = x2 - 2*x1 + x0
            r_norm = norm(r)
            v_norm = norm(v)
            if not np.isfinite(r_norm) and not np.isfinite(v_norm):
                msg = 'failed to compute step size: neither norm is finite'
                raise _ConvergenceError(msg)
            if not v_norm:
                msg = 'failed to compute step size: v_norm is zero'
                raise _ConvergenceError(msg)
            a = -r_norm / v_norm
            step = functools.partial(_step, x0, r, v)
            if L is not None:
                a = _modify_step_length(a, L, backtrack_rate, step)
            xp = step(a)
            xn = func(xp, *args)
            completed_iterations += 1
            if norm(xn - xp) < atol:
                return OptimizeResult(x=xn, success=True, status=0,
                        message='Success', nfev=func.ncalls,
                        nit=completed_iterations)
            if maxiter is not None and completed_iterations >= maxiter:
                raise _ConvergenceError('exceeded %d iterations' % maxiter)
            x0 = xn
    except _ConvergenceError as e:
        return OptimizeResult(x=xn, success=False, status=1,
                message=str(e), nfev=func.ncalls, nit=completed_iterations)

