"""
Loss function for solving MMPDE Problem
"""
import jax 
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Any, Callable, NamedTuple

Points = ArrayLike
TransFn = Callable[[Points], Points]
MonitorFn = Callable[[Points], float]

def loss_MMPDE(Xc:Points, tranFn:TransFn, monitorFn:MonitorFn)->float:
    """Calculate MMPDE loss (Eq. 17)

    Args:
        Xc (Points): Sampling points in computational domain
        tranFn (TransFn): Transformation function from computational domain to physical domain

    Returns:
        float: loss value
    """
    pass

def diff(Xc, tranFn, monitorFn):
    mJ_func = jax.jacfwd(tranFn)

    j = jnp.linalg.det(mJ_func(Xc))
    w = monitorFn(Xc)
    
    def s12(Xc):
        mJ = mJ_func(Xc)
        xy_xi = mJ[0,:]
        xy_eta = mJ[1,:]
        output = xy_xi * xy_eta.T
        output = output / (j*w)
        return xy_xi, xy_eta
    


def S1(Xc, tranFn, monitorFn)->float:
    """Calculate S1 (Eq. 15)

    Args:
        xi (float): xi value
        eta (float): eta value
        tranFn (TransFn): Transformation function from computational domain to physical domain

    Returns:
        float: S1 value
    """
    pass