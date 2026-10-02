"""
    AbstractBoundsHandling

How a solver treats the `lb`/`ub` of a problem when it has no bound-constrained method of
its own. A solver takes one as its `bounds_handling` keyword. Subtypes define

  - [`handles_bounds_natively`](@ref): `true` if the solver works in the original
    coordinates and keeps its iterates in the box itself, so that `SciMLBase.allowsbounds`
    holds and the change of variables is skipped.
  - [`projects_iterates`](@ref): `true` if every candidate iterate is clamped into the box.

Available options:

  - [`BoundsTransform`](@ref): solve the unconstrained problem obtained by a change of
    variables (the default).
  - [`BoundsProjection`](@ref): solve in the original coordinates and clamp every iterate.
"""
abstract type AbstractBoundsHandling end

"""
    BoundsTransform()

Handle bounds with a change of variables that maps the box onto the real line (the default).
A root on a bound is reached only in the limit.
"""
struct BoundsTransform <: AbstractBoundsHandling end

"""
    BoundsProjection()

Handle bounds by clamping every candidate iterate into `[lb, ub]` before the residual is
evaluated. A root on a bound is reached exactly and the residual is never called outside the
box. This is a projected method, not an active-set method, and can stall at a corner of the
box.
"""
struct BoundsProjection <: AbstractBoundsHandling end

"""
    handles_bounds_natively(handling::AbstractBoundsHandling)

Whether a solver given `handling` works in the original coordinates itself.
"""
handles_bounds_natively(::AbstractBoundsHandling) = false
handles_bounds_natively(::BoundsProjection) = true

"""
    projects_iterates(handling::AbstractBoundsHandling)

Whether a solver given `handling` clamps every candidate iterate into the box.
"""
projects_iterates(::AbstractBoundsHandling) = false
projects_iterates(::BoundsProjection) = true
