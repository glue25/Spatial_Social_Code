"""Closed-loop Spatial--Social Code experiment support.

This package is deliberately independent of EnactToM's evaluator.  Runtime
state is updated only from an agent's private instruction, action feedback,
and messages that the environment actually delivered.
"""

from .runtime import ClosedLoopRuntime

__all__ = ["ClosedLoopRuntime"]
