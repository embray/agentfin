# -*- coding: utf-8 -*-

import numpy as np
from plotting import plot_price, plot_returns

# A Nonlinear Structural Model for Volatility Clustering
# This program is designed to duplicate the results found by
# Andrea Gaunersdorfer & Cars Hommes
# It also builds off the earlier results from
# Gaunersdorfer, Hommes, Wagener, Journal of Economic Behavior and
# Organization, Vol 67, 27-47: 2008

# Type 1 agents hold fundamentalist beliefs
# Type 2 agents are trend followers
# E1t[pt+1] = p* + v(pt-1 - p*)
# E2t[pt+1] = pt-1 + g(pt-1 - pt-2)


# α*σ (risk aversion * variance)
# Note: Because Python 3 supports most unicode characters in variable names we
# could actually name this variable:
#
# ασ = 1
#
# Though it would probably be less convenient to type.  Similarly with ȳ = 1.
#
# For use in this simulation, these values are assumed constant 1, and won't
# generally change.  If there is a reason to change them we could make them
# additional paramters of the simulation, but for the purpose of demonstration
# we will make them global "constants", which in Python are typically written
# in ALL_CAPS to distinguish that they are "constant" (of course the Python
# language does not actually prohibit us from changing them).
A_SIG = 1.0
Y_BAR = 1.0


# Default parameters without chaos (chaos=False)
GH_DEFAULTS = dict(
    T = 10000,
    r = 0.001,
    beta = 2,
    v = 1.0,
    g = 1.9,
    alpha = 1800,
    eta = 0.99,
    risk_adjust = 0,
    eps_sig = 10,
    start_x = -400
)


# Default parameters with chaos
GH_DEFAULTS_CHAOS = dict(
    T = 500,
    r = 0.01,
    beta = 4,
    v = 0.3,
    # g == 2.00 # fixed point
    # g = 2.09 # limit cycle
    g = 2.4,  # chaos
    eta = 0,
    # std for price noise
    eps_sig = 0,
    alpha = 10,
    # set to one for risk adjustment trading profits(done differnently in the
    # two papers)
    risk_adjust = 1,
    # starting price level
    start_x = 0.01
)


def gaunersdorfer_hommes(chaos=False, **kwargs):
    """
    Gaunersdorfer and Hommes model.

    All parameters are optional and have default values set below.  The default
    value for each parameter depends on whether or not ``chaos=True``.

    For GHW model with low D chaos set ``chaos=True``.
    For GH model with limit cycle and realistic price dynamics set
    ``chaos=False``.

    The model is written with the dynamics in terms of deviations
    from the constant fundamental given by pbar
    as x(t) = p(t)-pbar
    this is both a little simpler, and turns out to be a better
    thing to do numerically.  See notes for some details.

    T = Time Horizon
    """

    if chaos:
        # Important to call .copy() or else we will accidentally modify the
        # original default dicts below
        params = GH_DEFAULTS_CHAOS.copy()
    else:
        params = GH_DEFAULTS.copy()

    # If the user passed any additional arguments as keyword arguments, we
    # replace the defaults with the user's arguments here.
    params.update(kwargs)

    # Now read out local variables from the defaults (as a trick, we could also
    # do this in one line by running `locals().update(params)`.  `locals()`
    # returns the dictionary that actually stores the local variables for our
    # function, and we can directly update it just like any other dict.
    # However, this can also be a bit dangerous (what if a user passes a
    # parameter that accidentally overrides a local variable that was not
    # intended to be passed as a parameter?)
    T = params['T']
    r = params['r']
    beta = params['beta']
    v = params['v']
    g = params['g']
    eta = params['eta']
    eps_sig = params['eps_sig']
    alpha = params['alpha']
    risk_adjust = params['risk_adjust']
    start_x = params['start_x']

    p_star = Y_BAR / r

    # Now we can set up the rest of the simulation from the given parameters:
    n2 = np.zeros([T, 1])       # Fraction of type 2
    p = np.zeros([T, 1])        # Price
    R = np.zeros([T, 1])         # Per share Return
    # x = deviation from fundamental (p_star)
    x = np.zeros([T, 1])         # p - p_star (used in most dynamics)
    u1 = np.zeros([T, 1])        # type 1 accumulated realized profits
    u2 = np.zeros([T, 1])       # type 2 accumulated realized profits
    z1 = 0.5 * np.ones([T, 1])   # share holdings type 1
    z2 = 0.5 * np.ones([T, 1])   # share holdings type 2
    eps = eps_sig * np.random.randn(T) # additive pricing noise
    nn = np.zeros([T,1])

    # initial value

    # fraction of agents
    n2[0] = 0.5
    n2[1] = n2[0]
    n2[2] = n2[1]
    n2[3] = n2[2]

    # x = price - p_star
    x[0] = start_x
    x[1] = x[0]
    x[2] = x[1]
    x[3] = x[2]

    p = p_star + x

    # start at t = 5 to allow for lags
    for t in range(4, T):
        # update utility
        # simplified equation from paper(see GHW equation(12))
        # u1(t) = -0.5*(x(t-1)-v*x(t-3))^2;
        # u2(t) = -0.5*(x(t-1) - x(t-3) - g*(x(t-3)-x(t-4)))^2;
        # detaled one period profits using last period holdings
        pi1 = R[t-1] * z1[t-2] - risk_adjust * 0.5 * A_SIG * z1[t-2]**2
        pi2 = R[t-1] * z2[t-2] - risk_adjust * 0.5 * A_SIG * z2[t-2]**2
        # accumulated fitness
        u1[t-1] = pi1 + eta * u1[t-2]
        u2[t-1] = pi2 + eta * u2[t-2]
        # normalization for logistice
        norm = np.exp(beta * u1[t-1]) + np.exp(beta * u2[t-1])
        nn[t] = norm
        # basic n2tilde (before adjustment)
        n2tilde = np.exp(beta * u2[t-1]) / norm
        # emergency check to make sure still in range, if not set to 0.5
        if np.isnan(n2tilde):
            n2tilde = 0.5
        # adjustment to n, see paper
        n2[t] = n2tilde * np.exp(-(x[t-1]) ** 2 / alpha)
        # x(t+1) ( p(t+1)) forecasts
        exp1 = v * (x[t-1]) # type 1 price forecast for t+1
        exp2 = x[t-1] + g * (x[t-1]-x[t-2]) # type 2 price forecast for t+1
        # new price for today from t+1 forecasts (note timing)
        x[t] = 1/(1+r) * (((1-n2[t])* exp1 + n2[t]*exp2 ) + eps[t])
        p[t] = x[t] + p_star
        # returns time path
        # R[t-1] = p[t+1] - p_star - (1+r)*(p[t]-p_star) + dstd*np.randn(1)
        R[t] = x[t] - x[t-1]
        # portfolio decisions
        z1[t] = (exp1 - x[t]) / A_SIG
        z2[t] = (exp2 - x[t]) / A_SIG

    # log return
    lret = np.log(p[1:T]) - np.log(p[0:T-1])
    # arithmetic return
    ret = p[1:T] / p[0:T-1]-1
    ghret = p[1:T] - Y_BAR - (1+r) * p[0:T-1]

    # We return the array of prices and arrays of returns.
    # We could also return other intermediate results as needed.
    # Some functions in scipy, for example, take additional arguments which
    # affect exactly which results are returned, and which ones are thrown
    # away.  Or others may take a single flag to optionally return extended
    # result parameters in a dict.
    # See for example scipy.optimize.least_squares for an example interface
    # with many inputs, and many non-trivial outputs:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
    return p, lret, ret, ghret


p, lret, ret, ghret = gaunersdorfer_hommes()

plot_price(p)
plot_returns(lret)
