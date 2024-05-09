"""
Loss function for solving MMPDE Problem
"""
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

