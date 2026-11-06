module SCCNonlinearSolve

import SciMLBase
import CommonSolve
import NonlinearSolveBase
import SymbolicIndexingInterface
import SciMLBase: NonlinearProblem, NonlinearLeastSquaresProblem, LinearProblem
using SciMLPublic: @public

"""
    SCCAlg(; nlalg = nothing, linalg = nothing, store_original = Val(false))

Algorithm for solving Strongly Connected Component (SCC) problems containing
both nonlinear and linear subproblems.

### Keyword Arguments

  - `nlalg`: Algorithm to use for solving NonlinearProblem components
  - `linalg`: Algorithm to use for solving LinearProblem components
  - `store_original`: Whether to store the individual sub-solutions in the
    `original` field of the returned solution. Default `Val(false)` to keep
    the return type simple (required for Enzyme AD compatibility). Set to
    `Val(true)` for debugging to inspect individual sub-solutions.
"""
struct SCCAlg{N, L, S <: Val}
    nlalg::N
    linalg::L
    store_original::S
end

function SCCAlg(; nlalg = nothing, linalg = nothing, store_original = Val(false))
    return SCCAlg(nlalg, linalg, store_original)
end

function CommonSolve.solve(
        prob::SciMLBase.SCCNonlinearProblem;
        sensealg = nothing, u0 = nothing, p = nothing, kwargs...
    )
    return CommonSolve.solve(prob, SCCAlg(); sensealg, u0, p, kwargs...)
end

function CommonSolve.solve(prob::SciMLBase.SCCNonlinearProblem, ::Nothing; kwargs...)
    return CommonSolve.solve(prob; kwargs...)
end

function CommonSolve.solve(
        prob::SciMLBase.SCCNonlinearProblem,
        alg::SciMLBase.AbstractNonlinearAlgorithm;
        sensealg = nothing, u0 = nothing, p = nothing, kwargs...
    )
    return CommonSolve.solve(prob, SCCAlg(; nlalg = alg); sensealg, u0, p, kwargs...)
end

function CommonSolve.solve(
        prob::SciMLBase.SCCNonlinearProblem, alg::SCCAlg;
        sensealg = nothing, u0 = nothing, p = nothing, kwargs...
    )
    # Note: SCCNonlinearProblem does not have a u0 field - each subproblem has its own u0.
    # The u0 parameter here is only used for AD hooks, not for actual solving.
    p = p !== nothing ? p : prob.p
    return scc_solve_up(prob, sensealg, u0, p, alg; kwargs...)
end

function CommonSolve.init(
        prob::SciMLBase.SCCNonlinearProblem; sensealg = nothing, u0 = nothing,
        p = nothing, kwargs...
    )
    return CommonSolve.init(prob, SCCAlg(); sensealg, u0, p, kwargs...)
end

function CommonSolve.init(prob::SciMLBase.SCCNonlinearProblem, ::Nothing; kwargs...)
    return CommonSolve.init(prob; kwargs...)
end

function CommonSolve.init(
        prob::SciMLBase.SCCNonlinearProblem,
        alg::SciMLBase.AbstractNonlinearAlgorithm;
        sensealg = nothing, u0 = nothing, p = nothing, kwargs...
    )
    return CommonSolve.init(prob, SCCAlg(; nlalg = alg); sensealg, u0, p, kwargs...)
end

function CommonSolve.init(
        prob::SciMLBase.SCCNonlinearProblem, alg::SCCAlg;
        sensealg = nothing, u0 = nothing, p = nothing, kwargs...
    )
    probs = map(_concrete_scc_problem, prob.probs)
    # Keep the caller's container eltype so homogeneous and heterogeneous
    # block vectors share one `_scc_solve`/`step!` instance, matching `scc_solve_up`.
    if probs isa AbstractVector && all(Base.Fix2(isa, eltype(prob.probs)), probs)
        probs = copyto!(similar(prob.probs), probs)
    end
    concrete_prob = SciMLBase.remake(prob; probs)
    return SCCNonlinearCache(concrete_prob, alg, kwargs, __scc_fresh_sols(concrete_prob.probs), 1)
end

# An empty, per-block solution container in exactly the representation the driver
# builds up to: a growing `Tuple` for tuple-form blocks, or a concretely-`Vector{ST}`-
# typed vector (via `__scc_vector_sol_type`) for vector-form blocks. `view`s of the
# latter, even with zero elements so far, type-match what `solve_single_scc` passes a
# block's `explicitfuns!` during a full `solve`, which matters for e.g. tuple-typed or
# `FunctionWrappers`-typed `explicitfuns!`.
__scc_fresh_sols(probs::Tuple) = ()
__scc_fresh_sols(probs::AbstractVector) = Vector{__scc_vector_sol_type(probs)}()

"""
Internal solve function that can be hooked by ChainRulesCore for AD.
"""
function scc_solve_up(
        prob::SciMLBase.SCCNonlinearProblem, sensealg, u0, p, alg::SCCAlg;
        kwargs...
    )
    probs = map(_concrete_scc_problem, prob.probs)
    # Keep the caller's container eltype so homogeneous and heterogeneous
    # block vectors share one `_scc_solve` instance.
    if probs isa AbstractVector && all(Base.Fix2(isa, eltype(prob.probs)), probs)
        probs = copyto!(similar(prob.probs), probs)
    end
    concrete_prob = SciMLBase.remake(prob; probs)
    return _scc_solve(concrete_prob, alg; kwargs...)
end

function _concrete_scc_problem(prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem})
    SciMLBase.specialization(prob.f) === SciMLBase.AutoDespecialize || return prob
    return NonlinearSolveBase.get_concrete_problem(prob)
end
_concrete_scc_problem(prob) = prob

probvec(prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem, SciMLBase.HomotopyProblem}) = prob.u0
probvec(prob::LinearProblem) = prob.b

iteratively_build_sols(alg, sols; kwargs...) = sols

function solve_single_scc(alg, prob, explicitfun, sols; kwargs...)
    SciMLBase.invoke_with_despecialized_parameters(
        explicitfun, (prob.p, sols)
    )

    _sol = if prob isa SciMLBase.LinearProblem
        A = prob.A
        b = prob.b
        # `remake` to recalculate `A` and `b` based on updated parameters from `explicitfun`.
        # Pass `A` and `b` to avoid unnecessarily copying them.
        sol = SciMLBase.solve(SciMLBase.remake(prob; A, b), alg.linalg; kwargs...)
        # LinearSolution may have resid=nothing, so compute it: resid = A*u - b
        resid = isnothing(sol.resid) ? A * sol.u - b : sol.resid
        nlprob = NonlinearProblem{true}(Returns(nothing), sol.u, prob.p)
        SciMLBase.strip_solution(
            SciMLBase.build_solution(nlprob, nothing, sol.u, resid; sol.retcode)
        )
    else
        # A `HomotopyProblem` block (e.g. a Modelica `homotopy` operator block from
        # ModelingToolkit) is solved by continuation, not by the SCC's nonlinear algorithm
        # directly: that algorithm would only solve the block's target-λ system (see
        # `solve(::HomotopyProblem, ::AbstractNonlinearSolveAlgorithm)`). It is threaded in
        # as the *inner corrector* instead, so its autodiff / linear-solver choices are
        # honored while the block is swept from `simplified` to `actual`.
        blockalg = if prob isa SciMLBase.HomotopyProblem
            NonlinearSolveBase.HomotopyPolyAlgorithm(; inner = alg.nlalg)
        else
            alg.nlalg
        end
        sol = SciMLBase.solve(prob, blockalg; kwargs...)
        SciMLBase.strip_solution(
            SciMLBase.build_solution(
                prob, nothing, sol.u, sol.resid; sol.retcode
            )
        )
    end

    return _sol
end

# The stripped solution type produced by solving any block of `probs`, deterministic
# from the first problem: after `strip_solution`, all `NonlinearSolution`s have a
# predictable concrete type regardless of the original problem/algorithm types. Shared
# by the vector-form driver below and `SCCNonlinearCache` (`init`/`step!`/`reinit!`), so
# both always agree with each other and with however many type parameters
# `SciMLBase.NonlinearSolution` currently has.
function __scc_vector_sol_type(probs::AbstractVector)
    prob1 = first(probs)
    uType = typeof(probvec(prob1))
    T = eltype(uType)
    rType = uType  # resid has same type as u for nonlinear problems
    return SciMLBase.NonlinearSolution{
        T, 1, uType, rType,
        NamedTuple{(:p,), Tuple{Nothing}}, Nothing, Nothing, Nothing, Nothing, Nothing,
    }
end

function iteratively_build_sols(alg, probs::AbstractVector, explicitfuns::AbstractVector; kwargs...)
    ST = __scc_vector_sol_type(probs)
    sols = Vector{ST}(undef, length(probs))
    for i in eachindex(probs)
        sols[i] = solve_single_scc(alg, probs[i], explicitfuns[i], view(sols, 1:(i - 1)); kwargs...)::ST
    end
    return sols
end

@generated function iteratively_build_sols(alg, probs::Tuple, explicitfuns::Tuple, ::Val{N}; kwargs...) where {N}
    return quote
        Base.Cartesian.@nexprs $N i -> begin
            prob_i = solve_single_scc(
                alg, probs[i], explicitfuns[i],
                Base.Cartesian.@ntuple((i - 1), j -> prob_j); kwargs...
            )
        end
        return Base.Cartesian.@ntuple $N i -> prob_i
    end
end

function iteratively_build_sols(alg, sols, (prob, explicitfun), args...; kwargs...)
    p = SymbolicIndexingInterface.parameter_values(prob)
    SciMLBase.invoke_with_despecialized_parameters(
        explicitfun, (p, sols)
    )

    _sol = if prob isa SciMLBase.LinearProblem
        A = prob.A
        b = prob.b
        # `remake` to recalculate `A` and `b` based on updated parameters from `explicitfun`.
        # Pass `A` and `b` to avoid unnecessarily copying them.
        sol = SciMLBase.solve(SciMLBase.remake(prob; A, b), alg.linalg; kwargs...)
        # LinearSolution may have resid=nothing, so compute it: resid = A*u - b
        resid = isnothing(sol.resid) ? A * sol.u - b : sol.resid
        SciMLBase.build_linear_solution(
            alg.linalg, sol.u, resid, nothing; sol.retcode
        )
    else
        sol = SciMLBase.solve(prob, alg.nlalg; kwargs...)
        SciMLBase.build_solution(prob, nothing, sol.u, sol.resid; sol.retcode)
    end

    return iteratively_build_sols(alg, (sols..., _sol), args...; kwargs...)
end

# Combines the per-block solutions `sols` (in SCC order) into the single
# `NonlinearSolution` that `solve`/`solve!` return. Shared by `_scc_solve` (the
# all-at-once path) and `SCCNonlinearCache`'s `solve!` (the stepped path), so the
# two can only disagree if `solve_single_scc` itself is nondeterministic.
function __scc_combine_sols(prob::SciMLBase.SCCNonlinearProblem, alg::SCCAlg, sols)
    # Use splatted vcat instead of reduce (reduce over tuples infers Any,
    # which exceeds Enzyme's type analysis limits).
    u = vcat(map(s -> s.u, sols)...)
    resid = vcat(map(s -> s.resid, sols)...)

    # Simple loop instead of all/findfirst (higher-order functions over
    # tuples can confuse Enzyme's activity analysis).
    retcode = SciMLBase.ReturnCode.Success
    for s in sols
        if !SciMLBase.successful_retcode(s)
            retcode = s.retcode
            break
        end
    end

    # Only store sub-solutions in `original` when requested (Val(true)).
    # The sub-solutions tuple nested in the kwargs NamedTuple exceeds
    # Enzyme's type analysis depth, so it's off by default.
    original = alg.store_original isa Val{true} ? sols : nothing
    return SciMLBase.build_solution(prob, alg, u, resid; retcode, original)
end

"""
Internal solve implementation for SCCNonlinearProblem.
This is called by scc_solve_up and should NOT be hooked by ChainRulesCore.
"""
function _scc_solve(prob::SciMLBase.SCCNonlinearProblem, alg::SCCAlg; kwargs...)
    numscc = length(prob.probs)
    if prob.probs isa Tuple
        sols = iteratively_build_sols(alg, prob.probs, Tuple(prob.explicitfuns!), Val(numscc); kwargs...)
    else
        sols = iteratively_build_sols(alg, prob.probs, prob.explicitfuns!; kwargs...)
    end
    return __scc_combine_sols(prob, alg, sols)
end

"""
    SCCNonlinearCache

Cache returned by `init(prob::SCCNonlinearProblem, alg)`. An `SCCNonlinearProblem`
solves as an ordered sequence of block solves rather than through a single mutable
nonlinear solver state, so this cache tracks progress through that sequence
(which blocks are solved, and their solutions so far) instead of wrapping one
solver's internal iterate. `cache.sols` holds the per-block solutions recorded so
far, in the *same* container representation `solve` itself builds up internally
(a growing `Tuple` for tuple-form blocks; a concretely-typed, growing `Vector` for
vector-form blocks) — `length(cache.sols)` is the number of blocks solved so far,
and `cache.sols[i]`/indexing into it works exactly as the `sols` argument that a
block's own `explicitfuns!` receives.

### Interface

  - `step!(cache)` solves the next unsolved block: it applies that block's
    `explicitfuns!` against the solutions of every previously solved block, in
    the exact representation (tuple, or typed vector view) that `solve` itself
    passes, and records the result. Calling `step!` on a cache with no
    remaining blocks is a no-op.
  - `solve!(cache)` solves every remaining block and returns the same
    `NonlinearSolution` that `solve(prob, alg)` returns: identical `u`, `resid`,
    `retcode`, and (with `store_original = Val(true)`) `original` container
    type. This holds whether `solve!` is called right after `init` (no blocks
    solved yet), after some number of `step!` calls, after stepping to
    completion, or repeated again afterward — `solve!` never redoes a block
    `step!`/a prior `solve!` already solved, and repeated calls keep returning
    the same completed solution.
  - `SciMLBase.reinit!(cache, u0 = missing; p = missing)` discards every
    block's recorded solution and rewinds `step!`/`solve!` progress back to the
    first block. If `u0`/`p` are given, the underlying `SCCNonlinearProblem` is
    first `remake`d with them (see that method's docstring for how `u0`/`p`
    are chunked and distributed across blocks, and the `parameters_alias`
    requirement on `p`) before resetting progress on the updated problem.

Does not support symbolic indexing or state access mid-solve (consistent with
`SteadyStateDiffEq.jl`'s analogous `SteadyStateSCCCache`, which has the same
block-sequential structure and the same limitation).
"""
mutable struct SCCNonlinearCache{P, A, K}
    prob::P
    const alg::A
    const kwargs::K
    sols
    idx::Int
end

function SciMLBase.step!(cache::SCCNonlinearCache, args...; kwargs...)
    probs = cache.prob.probs
    cache.idx > length(probs) && return cache
    i = cache.idx
    sols_so_far = probs isa Tuple ? cache.sols : view(cache.sols, 1:(i - 1))
    _sol = solve_single_scc(
        cache.alg, probs[i], cache.prob.explicitfuns![i], sols_so_far;
        cache.kwargs..., kwargs...
    )
    if probs isa Tuple
        cache.sols = (cache.sols..., _sol)
    else
        push!(cache.sols, _sol)
    end
    cache.idx += 1
    return cache
end

function CommonSolve.solve!(cache::SCCNonlinearCache)
    probs = cache.prob.probs
    if cache.idx == 1 && isempty(cache.sols)
        # No stepping has happened yet: build the per-block solutions with the same
        # driver `solve(prob, alg)` uses (the `@generated` Cartesian unrolling for a
        # `Tuple` of blocks, in particular, is more specialized than looping `step!`),
        # so performance and results are identical to a direct `solve`. Retained on
        # the cache so a *second* `solve!` (or a completion no-op `step!` followed by
        # `solve!`) recombines the real solutions instead of an empty history.
        sols = if probs isa Tuple
            iteratively_build_sols(cache.alg, probs, Tuple(cache.prob.explicitfuns!), Val(length(probs)); cache.kwargs...)
        else
            iteratively_build_sols(cache.alg, probs, cache.prob.explicitfuns!; cache.kwargs...)
        end
        cache.sols = sols
        cache.idx = length(probs) + 1
        return __scc_combine_sols(cache.prob, cache.alg, sols)
    end
    while cache.idx <= length(probs)
        SciMLBase.step!(cache)
    end
    return __scc_combine_sols(cache.prob, cache.alg, cache.sols)
end

"""
    SciMLBase.reinit!(cache::SCCNonlinearCache, u0 = missing; p = missing)

Reset `cache` to an `init`-fresh state: discards every block's recorded
solution and rewinds `step!`/`solve!` progress back to the first block.

If `u0` and/or `p` are given (anything other than the default `missing`), the
underlying `SCCNonlinearProblem` is first replaced with
`SciMLBase.remake(cache.prob; u0, p)` — `u0` is the whole-problem state vector,
chunked and distributed across blocks; `p` updates every block's parameters,
and requires the problem's `parameters_alias` to be `true` (see
`remake(::SCCNonlinearProblem)`'s docstring) — before progress is reset on the
updated problem. Omitting both resets progress on the current problem
unchanged.
"""
function SciMLBase.reinit!(
        cache::SCCNonlinearCache, u0 = missing; p = missing
    )
    if u0 !== missing || p !== missing
        cache.prob = SciMLBase.remake(cache.prob; u0, p)
    end
    cache.sols = __scc_fresh_sols(cache.prob.probs)
    cache.idx = 1
    return cache
end

export scc_solve_up
@public SCCNonlinearCache

end
