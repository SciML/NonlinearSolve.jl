#
# The test fixture: a 1-D bar of `n_el` linear elements with one quadrature point each,
# carrying three internal variables per element. Written twice — in condensed multi-level
# form (`Rbar!` + `assemble_S!` + `commit_internal!`) and monolithically over the full
# `[ū; vec(Q)]` — so the two can be checked against each other.
#
#     ε_i = (u_i - u_{i-1}) / h                        (u_0 ≡ 0)
#     r_i = q_i - γ·tanh.(c_i·ε_i + D·q_i)             local, 3 equations per element
#     σ_i = E·(ε_i - m·q_i)
#     R̄_j = σ_j - σ_{j+1}   (j < n),   R̄_n = σ_n - f_ext
#
using MultiLevelNonlinearSolve
using LinearAlgebra, SparseArrays

const GAMMA = 0.7
const E_MOD = 1.0
const F_EXT = 0.3
const M_VEC = [0.4, 0.3, 0.2]
const C_VEC = [2.0, -3.0, 2.5]
# `D` is not free decoration. It is chosen so that
#   (a) `I - γ·Diagonal(sech²)·D` stays well conditioned (≤ 1.41 over the whole strain
#       range), and
#   (b) the tangent modulus only varies between 0.68 and 1.0, i.e. `S(ū₀)` is within a
#       factor 1.1 of `S(ū*)`.
# (b) is what makes the `:chord` runs converge at all: with a stronger coupling
# `Dt(ū₀)/Dt(ū*)` approaches 2 and the frozen-Jacobian contraction factor
# `|1 - Dt(ū*)/Dt(ū₀)|` exceeds 1. Any replacement fixture needs the same property.
const D_MAT = [0.25 0.10 -0.15; 0.10 0.20 0.10; -0.15 0.10 0.30]

mutable struct BarCounters
    nresidual::Int      # condensed residual evaluations
    nassembly::Int      # `assemble_S!` calls, including any the framework makes itself
    ncommit::Int        # commit steps
    nlocaliter::Int     # total local Newton iterations
    assembly_types::Vector{Any}
end
BarCounters() = BarCounters(0, 0, 0, 0, Any[])

"""
    BarModel(; kwargs...)

Parameters and workspaces of the bar. `buffer` double-buffers the internal variables, so a
trial residual never touches committed state; `ensemble` partitions the elements into chunks
for the threaded runs.

`corrector_scale` multiplies `dq/dε` inside `assemble_S!` only — 1.0 is the exact Schur
corrector, anything else is a deliberately wrong tangent. `cscale` scales `c` per element;
left at `ones`, the bar is homogeneous and every element has the same tangent.
"""
struct BarModel{B}
    n::Int
    h::Float64
    cscale::Vector{Float64}
    buffer::B
    ensemble::LocalEnsemble
    σ::Vector{Float64}
    Dt::Vector{Float64}
    chunk_ok::Vector{Bool}
    chunk_iters::Vector{Int}
    local_tol::Float64
    local_maxiter::Int
    corrector_scale::Float64
    fail_above::Float64
    threaded::Bool
    counters::BarCounters
end

function BarModel(;
        n_el = 40, cscale = ones(n_el), local_tol = 1.0e-14, local_maxiter = 50,
        corrector_scale = 1.0, fail_above = Inf, threaded = false,
        nchunks = Threads.nthreads()
    )
    ensemble = LocalEnsemble(n_el; nchunks)
    nc = length(ensemble.chunks)
    return BarModel(
        n_el, 1 / n_el, cscale, LocalStateBuffer(zeros(3, n_el)), ensemble,
        zeros(n_el), zeros(n_el), fill(true, nc), zeros(Int, nc),
        local_tol, local_maxiter, corrector_scale, fail_above, threaded, BarCounters()
    )
end

@inline strain(ū, model::BarModel, i::Int) =
    (ū[i] - (i == 1 ? zero(eltype(ū)) : ū[i - 1])) / model.h
@inline elem_c(model::BarModel, i::Int) = C_VEC .* model.cscale[i]

"""
    local_solve!(q, model, ε, c, tol) -> (iterations, residual)

Newton on `q - γ·tanh.(c·ε + D·q) = 0` at fixed `ε`, warm-started from the incoming `q`.

`model.fail_above` makes the local problem *unsolvable* beyond a strain threshold, which is
how the failure-semantics tests reach the "local solve diverged" branch — the analogue of a
return map falling outside its admissible region.
"""
function local_solve!(q, model::BarModel, ε, c, tol)
    abs(ε) > model.fail_above && return (0, Inf)
    res = Inf
    for it in 1:model.local_maxiter
        g = tanh.(c .* ε .+ D_MAT * q)
        r = q .- GAMMA .* g
        res = norm(r, Inf)
        res ≤ tol && return (it - 1, res)
        s = 1 .- g .^ 2
        q .-= (I - GAMMA .* (s .* D_MAT)) \ r
    end
    return (model.local_maxiter, res)
end

"`dq/dε = (I - γ·∂g/∂q) \\ (γ·∂g/∂ε)`, then `Dt = dσ/dε = E·(1 - m·dq/dε)`."
function element_tangent(model::BarModel, ε, q, c)
    g = tanh.(c .* ε .+ D_MAT * q)
    s = 1 .- g .^ 2
    dqdε = ((I - GAMMA .* (s .* D_MAT)) \ (GAMMA .* (s .* c))) .* model.corrector_scale
    return E_MOD * (1 - dot(M_VEC, dqdε))
end

"""
    solve_local_ensemble!(model, ū, tol) -> Bool

Run every local problem at `ū` into scratch state, warm-started from committed state, and
fill `model.σ`. Each chunk writes only its own points and its own reduction slot; the
reductions themselves are folded serially afterwards, so a threaded run is reproducible.
"""
function solve_local_ensemble!(model::BarModel, ū, tol)
    fill!(model.chunk_ok, true)
    fill!(model.chunk_iters, 0)
    ensemble_foreach(model.ensemble, ū, tol, model; threaded = model.threaded) do chunk,
            ichunk, ū, tol, model
        for i in chunk
            ε = strain(ū, model, i)
            q = trial_state(model.buffer, i)
            its, res = local_solve!(q, model, ε, elem_c(model, i), tol)
            model.chunk_iters[ichunk] += its
            res ≤ tol || (model.chunk_ok[ichunk] = false)
            model.σ[i] = E_MOD * (ε - dot(M_VEC, q))
        end
    end
    model.counters.nlocaliter += sum(model.chunk_iters)
    return all(model.chunk_ok)
end

"""
    Rbar!(res, ū, p)

The condensed residual. Every call is a trial: it reads committed internal state, writes only
scratch, and reports a diverged local ensemble as `Inf` rows (never `NaN`, which would make a
backtracking line search burn its whole iteration budget at a `NaN` state).
"""
function Rbar!(res, ū, p)
    model = user_parameters(p)
    model.counters.nresidual += 1
    tol = something(local_tolerance(p), model.local_tol)
    if !solve_local_ensemble!(model, ū, tol)
        fill!(res, Inf)
        return nothing
    end
    for j in 1:(model.n - 1)
        res[j] = model.σ[j] - model.σ[j + 1]
    end
    res[model.n] = model.σ[model.n] - F_EXT
    return nothing
end

"""
    assemble_S!(S, ū, p)

The Schur tangent. It runs no local solves: the committed internal state is guaranteed
consistent with `ū` here, so the per-element correctors are read straight off it.
"""
function assemble_S!(S, ū, p)
    model = user_parameters(p)
    model.counters.nassembly += 1
    push!(model.counters.assembly_types, typeof(S))
    for i in 1:model.n
        ε = strain(ū, model, i)
        model.Dt[i] = element_tangent(
            model, ε, committed_state(model.buffer, i), elem_c(model, i)
        )
    end
    fill!(S, 0)
    Dt, h, n = model.Dt, model.h, model.n
    for j in 1:(n - 1)
        S[j, j] = (Dt[j] + Dt[j + 1]) / h
        S[j, j + 1] = -Dt[j + 1] / h
        S[j + 1, j] = -Dt[j + 1] / h
    end
    S[n, n] = Dt[n] / h
    return nothing
end

"""
    commit_internal!(q_dest, ū, p)

The commit step: re-solve the locals at the accepted `ū`, promote them to committed, and
report whether they all converged.
"""
function commit_internal!(q_dest, ū, p)
    model = user_parameters(p)
    model.counters.ncommit += 1
    tol = something(local_tolerance(p), model.local_tol)
    ok = solve_local_ensemble!(model, ū, tol)
    commit_local_state!(model.buffer)
    copyto!(q_dest, vec(model.buffer.committed))
    return ok
end

"The same problem with nothing eliminated: `x = [ū; vec(Q)]`, length `4n`."
function monolithic!(F, x, p)
    model = user_parameters(p)
    n = model.n
    σ = Vector{eltype(x)}(undef, n)
    for i in 1:n
        ε = strain(x, model, i)
        rng = (n + 3 * (i - 1) + 1):(n + 3 * i)
        q = view(x, rng)
        σ[i] = E_MOD * (ε - dot(M_VEC, q))
        F[rng] .= q .- GAMMA .* tanh.(elem_c(model, i) .* ε .+ D_MAT * q)
    end
    for j in 1:(n - 1)
        F[j] = σ[j] - σ[j + 1]
    end
    F[n] = σ[n] - F_EXT
    return nothing
end

sparse_prototype(n::Int) = spdiagm(-1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1))

"""
    bar_problem(; jac = assemble_S!, kwargs...)

The full multi-level problem and its model, ready for `solve`/`init`. `jac` is a hook for the
tests that spy on the assembler.
"""
function bar_problem(; jac = assemble_S!, local_tolerance = nothing, u0 = nothing, kwargs...)
    model = BarModel(; kwargs...)
    n = model.n
    f = MultiLevelNonlinearFunction(
        NonlinearFunction(Rbar!; jac, jac_prototype = sparse_prototype(n));
        primary = 1:n, internal = (n + 1):(4n), commit_internal!, local_tolerance
    )
    return NonlinearProblem(f, u0 === nothing ? zeros(4n) : u0, model), model
end

"The monolithic twin of `bar_problem`, over the same full state."
function monolithic_problem(model::BarModel)
    return NonlinearProblem(NonlinearFunction(monolithic!), zeros(4 * model.n), model)
end

"Observed order of each residual triple: `log(e_{k+1}/e_k) / log(e_k/e_{k-1})`."
observed_orders(e) = [log(e[k + 1] / e[k]) / log(e[k] / e[k - 1]) for k in 2:(length(e) - 1)]

"""
    residual_history(cache; kwargs...)

Drive `cache` one `step!` at a time, recording `‖R̄‖_∞` per global iteration.
`chord_after` leading steps recompute the Jacobian; later steps pass
`recompute_jacobian = false`.
"""
function residual_history(cache; chord_after = 0, maxsteps = 200)
    primary = cache.prob.f.primary
    e = [norm(view(NonlinearSolveBase.get_fu(cache), primary), Inf)]
    k = 0
    while NonlinearSolveBase.not_terminated(cache) && k < maxsteps
        k += 1
        chord_after == 0 ? step!(cache) :
            step!(cache; recompute_jacobian = k <= chord_after)
        push!(e, norm(view(NonlinearSolveBase.get_fu(cache), primary), Inf))
    end
    return e
end

"""
    tail_order(e; floor = 1e-14)

Observed order over the last triple whose three residuals are all above the roundoff floor.
Fitting the floored tail instead would report a linear rate no matter what the method does.
"""
function tail_order(e; floor = 1.0e-14)
    q = observed_orders(e)
    isempty(q) && return NaN
    clean = findall(k -> e[k + 2] > floor, 1:(length(e) - 2))
    return q[isempty(clean) ? length(q) : last(clean)]
end

"""
    is_superlinear(e; floor = 1e-14)

Whether the residual ratios `e_{k+1}/e_k` strictly decrease over the pre-floor segment.

This is the distinction that survives on a fixture converging in four iterations: a fitted
order needs an asymptotic window this problem never has, but a linear method (frozen
Jacobian) holds its ratio constant while a superlinear one keeps shrinking it.
"""
function is_superlinear(e; floor = 1.0e-14)
    keep = findall(>(floor), e)
    idx = first(keep):min(last(keep), length(e) - 1)
    length(idx) < 3 && return false
    r = [e[k + 1] / e[k] for k in idx]
    return all(r[k + 1] < r[k] for k in 1:(length(r) - 1))
end

const HETEROGENEOUS_CSCALE = [0.7 + 0.6 * (i - 1) / 39 for i in 1:40]
