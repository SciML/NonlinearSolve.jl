"""
    SimpleNonlinearSolve

Small, dependency-light nonlinear solvers.

This subpackage provides straightforward nonlinear solver algorithms such as
`SimpleNewtonRaphson`, `SimpleBroyden`, `SimpleTrustRegion`, and `SimpleDFSane`.
They are intended for direct use on small problems, scalar problems, and allocation
sensitive contexts where the full NonlinearSolve.jl stack is unnecessary.

### Example

```julia
using SimpleNonlinearSolve, SciMLBase

prob = NonlinearProblem((u, p) -> u^2 - p, 1.0, 2.0)
sol = solve(prob, SimpleNewtonRaphson())
```
"""
module SimpleNonlinearSolve

using ConcreteStructs: @concrete
using PrecompileTools: @compile_workload, @setup_workload
using Reexport: @reexport
using Setfield: @set!

using BracketingNonlinearSolve: BracketingNonlinearSolve
using CommonSolve: CommonSolve, solve, init, solve!
using LineSearch: AbstractLineSearchAlgorithm
using MaybeInplace: @bb
using NonlinearSolveBase: NonlinearSolveBase, L2_NORM,
    nonlinearsolve_forwarddiff_solve, nonlinearsolve_dual_solution,
    AbstractNonlinearSolveAlgorithm, NonlinearVerbosity
using SciMLBase: SciMLBase, NonlinearFunction, NonlinearProblem,
    NonlinearLeastSquaresProblem, ImmutableNonlinearProblem, ReturnCode, remake
using SciMLLogging: @SciMLMessage, AbstractVerbosityPreset
using LinearAlgebra: LinearAlgebra, dot

using StaticArraysCore: StaticArray, SArray, SVector, MArray

# AD Dependencies
using ADTypes: ADTypes, AutoForwardDiff
using DifferentiationInterface: DifferentiationInterface
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff, Dual

const DI = DifferentiationInterface

const DualNonlinearProblem = NonlinearProblem{
    <:Union{Number, <:AbstractArray}, iip,
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
} where {iip, T, V, P}

const DualNonlinearLeastSquaresProblem = NonlinearLeastSquaresProblem{
    <:Union{Number, <:AbstractArray}, iip,
    <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}},
} where {iip, T, V, P}

abstract type AbstractSimpleNonlinearSolveAlgorithm <: AbstractNonlinearSolveAlgorithm end
configure_autodiff(prob, alg::AbstractSimpleNonlinearSolveAlgorithm) = alg

const NLBUtils = NonlinearSolveBase.Utils

is_extension_loaded(::Val) = false

include("utils.jl")

include("broyden.jl")
include("dfsane.jl")
include("halley.jl")
include("klement.jl")
include("lbroyden.jl")
include("raphson.jl")
include("trust_region.jl")
include("simple_homotopy_sweep.jl")

# By Pass the highlevel checks for NonlinearProblem for Simple Algorithms
function CommonSolve.solve(
        prob::NonlinearProblem, alg::AbstractSimpleNonlinearSolveAlgorithm, args...;
        kwargs...
    )
    if prob.u0 === nothing
        return NonlinearSolveBase.build_null_solution(prob, args...; kwargs...)
    end
    prob = convert(ImmutableNonlinearProblem, prob)
    return solve(prob, alg, args...; kwargs...)
end

function CommonSolve.solve(
        prob::DualNonlinearProblem, alg::AbstractSimpleNonlinearSolveAlgorithm,
        args...; kwargs...
    )
    alg = configure_autodiff(prob, alg)
    prob = convert(ImmutableNonlinearProblem, prob)
    sol, partials = nonlinearsolve_forwarddiff_solve(prob, alg, args...; kwargs...)
    dual_soln = nonlinearsolve_dual_solution(sol.u, partials, prob.p)
    return SciMLBase.build_solution(
        prob, alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original
    )
end

function CommonSolve.solve(
        prob::DualNonlinearLeastSquaresProblem, alg::AbstractSimpleNonlinearSolveAlgorithm,
        args...; kwargs...
    )
    alg = configure_autodiff(prob, alg)
    sol, partials = nonlinearsolve_forwarddiff_solve(prob, alg, args...; kwargs...)
    dual_soln = nonlinearsolve_dual_solution(sol.u, partials, prob.p)
    return SciMLBase.build_solution(
        prob, alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original
    )
end

function CommonSolve.solve(
        prob::Union{ImmutableNonlinearProblem, NonlinearLeastSquaresProblem},
        alg::AbstractSimpleNonlinearSolveAlgorithm,
        args...; sensealg = nothing, u0 = nothing, p = nothing,
        initializealg = SciMLBase.NoInit(), kwargs...
    )
    if prob.u0 === nothing
        return NonlinearSolveBase.build_null_solution(prob, args...; kwargs...)
    end
    alg = configure_autodiff(prob, alg)
    cache = SciMLBase.__init(prob, alg, args...; initializealg, kwargs...)
    prob = cache.prob
    if cache.retcode == ReturnCode.InitialFailure
        return SciMLBase.build_solution(
            prob, alg, prob.u0,
            NonlinearSolveBase.Utils.evaluate_f(prob, prob.u0); cache.retcode
        )
    end
    if sensealg === nothing && haskey(prob.kwargs, :sensealg)
        sensealg = prob.kwargs[:sensealg]
    end
    new_u0 = u0 !== nothing ? u0 : prob.u0
    new_p = p !== nothing ? p : prob.p
    return simplenonlinearsolve_solve_up(
        prob, sensealg,
        new_u0, u0 === nothing,
        new_p, p === nothing,
        alg, args...;
        prob.kwargs..., kwargs...
    )
end

function simplenonlinearsolve_solve_up(
        prob::Union{ImmutableNonlinearProblem, NonlinearLeastSquaresProblem}, sensealg, u0,
        u0_changed, p, p_changed, alg, args...; kwargs...
    )
    (u0_changed || p_changed) && (prob = remake(prob; u0, p))
    return SciMLBase.__solve(prob, alg, args...; kwargs...)
end

@setup_workload begin
    for T in (Float64,)
        prob_scalar = NonlinearProblem{false}((u, p) -> u .* u .- p, T(0.1), T(2))
        prob_iip = NonlinearProblem{true}((du, u, p) -> du .= u .* u .- p, ones(T, 3), T(2))
        prob_oop = NonlinearProblem{false}((u, p) -> u .* u .- p, ones(T, 3), T(2))

        # Only compile frequently used algorithms -- mostly from the NonlinearSolve default
        #!format: off
        algs = [
            SimpleBroyden(),
            # SimpleDFSane(),
            SimpleKlement(),
            # SimpleLimitedMemoryBroyden(),
            # SimpleHalley(),
            SimpleNewtonRaphson(; autodiff = AutoForwardDiff(; chunksize = 1)),
            # SimpleTrustRegion()
        ]
        #!format: on

        @compile_workload begin
            @sync for prob in (prob_scalar, prob_iip, prob_oop), alg in algs

                Threads.@spawn CommonSolve.solve(prob, alg; abstol = 1.0e-2, verbose = false)
            end
        end
    end
end

# Rexexports
@reexport using ADTypes, SciMLBase, BracketingNonlinearSolve, NonlinearSolveBase

export SimpleBroyden, SimpleKlement, SimpleLimitedMemoryBroyden
export SimpleDFSane
export SimpleGaussNewton, SimpleNewtonRaphson, SimpleTrustRegion
export SimpleHalley
export SimpleHomotopySweep

export solve

end
