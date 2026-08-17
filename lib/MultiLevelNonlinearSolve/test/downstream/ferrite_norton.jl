#
# Norton viscoelasticity on a Ferrite mesh, solved with BOTH multi-level variants.
#
# ── What this is ────────────────────────────────────────────────────────────────────────
#
# A quasi-static solid whose material carries an internal variable (viscous strain) at every
# quadrature point. This is the problem shape multi-level Newton exists for: the displacement
# dofs `ū` are globally coupled through the mesh, while the internal variables `q` satisfy
# their own small nonlinear system at each quadrature point and are coupled to nothing but
# the strain there.
#
# ── Time discretisation ─────────────────────────────────────────────────────────────────
#
# The material law is a rate equation `dq/dt = g(ε(ū), q)`. Driving it from OrdinaryDiffEq is
# a later phase, so the time loop here is hand-rolled: fixed-Δt implicit (backward) Euler,
# which turns each time step into one *stationary* nonlinear problem
#
#     q = q_ref + γ̃·g(ε(ū), q),      γ̃ = Δt,   q_ref = q at the previous accepted step
#
# together with equilibrium `∫ ∇ˢδu : σ(ε(ū), q) dΩ = 0`. That local form is the canonical
# stage problem of a DIRK stage with γ = 1; a real integrator would supply `γ̃ = γ·Δt` and a
# stage-dependent `q_ref`, and nothing else about the callbacks below would change.
#
# ── The material (ported from FerriteDiffEq.jl's `PowerLawViscosityModel`) ──────────────
#
# A linear elastic spring in parallel with a Maxwell branch whose dashpot follows a Norton
# power law — a standard linear solid with nonlinear viscosity:
#
#     ℂ      = ν/((1+ν)(1−2ν))·I⊗I + 1/(1+ν)·𝐈ˢ            (Poisson-scaled unit tensor)
#     e      = ε − εᵛ                                       (overstrain; drives the dashpot)
#     σ      = E₀·ℂ:ε + E₁·ℂ:e                              (equilibrium + overstress branch)
#     dεᵛ/dt = (E₁/η₁)·(‖e‖/e₀)^(m−1)·ℂ:e                   (Norton power law, m = 1 ⇒ Maxwell)
#
# `q = Mandel(εᵛ)`, six components per quadrature point. The nonlinearity is the point: the
# local solve genuinely iterates and the tangents are state-dependent.
#
# The rate is evaluated as `(‖e‖²/e₀²)^((m−1)/2)` rather than `(‖e‖/e₀)^(m−1)`. For the m = 3
# used here the two are identical polynomials, but the squared form has no `sqrt` and so stays
# differentiable at `e = 0` — which the very first step, where `ε = q = 0`, walks straight
# into.
#
# ── Boundary conditions ─────────────────────────────────────────────────────────────────
#
# The standard Ferrite Newton pattern, and it is part of what this file demonstrates:
#
#   1. once per time step, `update!(ch, t)` then `apply!(ū, ch)` — the prescribed values are
#      written into the iterate, so it starts out satisfying them exactly;
#   2. in the residual, `apply_zero!(r, ch)` — the constrained rows are zeroed, so the Newton
#      correction leaves them alone;
#   3. in the tangent, `apply_zero!(S, scratch, ch)` — the constrained rows and columns are
#      zeroed with a unit diagonal, giving `δū = 0` there.
#
# Constrained dofs are displacement dofs by construction, so they all live in `ū`; the
# internal block is never constrained.
#
# The monolithic reference at the end departs from this in one place, and says why: a solver
# that differentiates the residual needs the constrained rows to carry `u - u_prescribed`
# rather than a zero, or their Jacobian rows come out empty.
#
using MultiLevelNonlinearSolve, Test
using Ferrite, Tensors
using ADTypes: AutoForwardDiff
using StaticArrays: SVector, SMatrix
using LinearAlgebra, SparseArrays

include("../convergence_helpers.jl")

# ── Material ────────────────────────────────────────────────────────────────────────────

struct NortonModel{T}
    E₀::T
    E₁::T
    η₁::T
    m::T
    e₀::T
    C6::SMatrix{6, 6, T, 36}   # ℂ in Mandel form; the whole local problem is 6-vector algebra
end

function NortonModel(; E₀ = 70.0e3, E₁ = 20.0e3, η₁ = 1.0e3, m = 3.0, e₀ = 2.0e-3, ν = 0.3)
    I2 = one(SymmetricTensor{2, 3})
    ℂ = ν / ((ν + 1) * (1 - 2ν)) * I2 ⊗ I2 + 1 / (1 + ν) * one(SymmetricTensor{4, 3})
    return NortonModel(E₀, E₁, η₁, m, e₀, SMatrix{6, 6}(tomandel(ℂ)))
end

"Viscous strain rate `g(ε, q)` in Mandel components."
function rate(model::NortonModel, ε6, q)
    e = SVector{6}(ε6) .- SVector{6}(q)
    fac = (model.E₁ / model.η₁) * (dot(e, e) / model.e₀^2)^((model.m - 1) / 2)
    return fac .* (model.C6 * e)
end

"Cauchy stress in Mandel components."
stress(model::NortonModel, ε6, q) =
    model.C6 * (model.E₀ .* SVector{6}(ε6) .+ model.E₁ .* (SVector{6}(ε6) .- SVector{6}(q)))

"""
    local_tangents(model, ε6, q) -> (∂g∂q, ∂g∂ε, ∂σ∂q, ∂σ∂ε)

All four 6×6 Mandel blocks of the quadrature-point response. `∂σ∂ε` is the frozen-`q`
tangent; the Schur corrector below turns it into the effective one.
"""
function local_tangents(model::NortonModel, ε6, q)
    C6, k = model.C6, model.E₁ / model.η₁
    e = SVector{6}(ε6) .- SVector{6}(q)
    s2 = dot(e, e)
    fac = k * (s2 / model.e₀^2)^((model.m - 1) / 2)
    # d(fac)/de = (m−1)·fac/‖e‖² · e, written without the division so it is finite at e = 0.
    dfac = s2 > 0 ? ((model.m - 1) * fac / s2) .* e : zero(e)
    ∂g∂ε = fac .* C6 .+ (C6 * e) * dfac'
    return (-∂g∂ε, ∂g∂ε, -model.E₁ .* C6, (model.E₀ + model.E₁) .* C6)
end

# ── Discretisation ──────────────────────────────────────────────────────────────────────

"""
    NortonProblem(; nel, Δt, ...)

Mesh, dof handler, constraints and every workspace the callbacks need.

`ū` is the Ferrite displacement field; `q` is a plain trailing block of `6 · n_qp` entries
indexed by global quadrature point, so the full state is `[ū; vec(Q)]`. The internal
variables need no Ferrite field of their own — nothing ever interpolates them.
"""
struct NortonProblem{DH, CH, CV, B}
    dh::DH
    ch::CH
    model::NortonModel{Float64}
    cellvalues::Vector{CV}          # one per chunk: `reinit!` mutates it
    ue::Vector{Vector{Float64}}     # per-chunk element dof scratch
    cdofs::Vector{Vector{Int}}      # per-chunk `celldofs!` scratch
    ccoords::Vector{Vector{Vec{3, Float64}}}
    buffer::B                       # committed/trial internal state, 6 × n_qp
    ensemble::LocalEnsemble         # partition of the CELLS (a cell owns `nqp` points)
    q_ref::Vector{Float64}          # internal state at the previous accepted time step
    σ::Vector{SymmetricTensor{2, 3, Float64, 6}}          # per quadrature point
    Ceff::Vector{SymmetricTensor{4, 3, Float64, 36}}      # per quadrature point
    chunk_ok::Vector{Bool}
    chunk_iters::Vector{Int}
    nqp::Int
    n_primary::Int
    n_internal::Int
    Δt::Float64
    Sproto::SparseMatrixCSC{Float64, Int}
    bc_values::Vector{Float64}
    local_tol::Float64
    local_maxiter::Int
    nassembly::Base.RefValue{Int}
    # What the last ensemble sweep solved. `commit_internal!` and `assemble_S!` are both
    # called at an iterate the residual has just been evaluated at, so without this the
    # ensemble runs three times per accepted iterate instead of once.
    last_key::Vector{Float64}          # the ū it ran at, plus the tolerance it ran at
    last_valid::Base.RefValue{Bool}
end

function NortonProblem(;
        nel = (2, 2, 2), Δt = 0.01, model = NortonModel(),
        nchunks = min(Threads.nthreads(), prod(nel)), local_tol = 1.0e-12,
        local_maxiter = 30, amplitude = 5.0e-3, period = 0.24
    )
    grid = generate_grid(Hexahedron, nel)
    ip = Lagrange{RefHexahedron, 1}()^3
    qr = QuadratureRule{RefHexahedron}(2)

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> (0.0, 0.0, 0.0)))
    # Sinusoidal stretch rather than a ramp: a monotone ramp cannot tell a step that uses the
    # wrong time from one that uses the right one.
    add!(
        ch, Dirichlet(
            :u, getfacetset(grid, "right"),
            (x, t) -> (amplitude * sinpi(2t / period), 0.0, 0.0)
        )
    )
    close!(ch)

    ncells = getncells(grid)
    nqp = getnquadpoints(QuadratureRule{RefHexahedron}(2))
    n_primary = ndofs(dh)
    n_internal = 6 * ncells * nqp
    ensemble = LocalEnsemble(ncells; nchunks)
    nc = length(ensemble.chunks)

    return NortonProblem(
        dh, ch, model,
        [CellValues(qr, ip) for _ in 1:nc],
        [zeros(ndofs_per_cell(dh, 1)) for _ in 1:nc],
        [zeros(Int, ndofs_per_cell(dh, 1)) for _ in 1:nc],
        [getcoordinates(grid, 1) for _ in 1:nc],
        LocalStateBuffer(zeros(6, ncells * nqp)), ensemble,
        zeros(n_internal),
        zeros(SymmetricTensor{2, 3, Float64, 6}, ncells * nqp),
        zeros(SymmetricTensor{4, 3, Float64, 36}, ncells * nqp),
        fill(true, nc), zeros(Int, nc), nqp, n_primary, n_internal, Δt,
        Ferrite.allocate_matrix(dh), zeros(length(ch.prescribed_dofs)), local_tol,
        local_maxiter, Ref(0), fill(NaN, n_primary + 1), Ref(false)
    )
end

@inline qpindex(p::NortonProblem, cellid, qp) = (cellid - 1) * p.nqp + qp

# ── The local problem: `q = q_ref + Δt·g(ε, q)` at one quadrature point ─────────────────

function solve_qp!(q, p::NortonProblem, ε6, q_ref)
    # Static all the way through: this is the kernel a real material model would put here, and
    # a per-quadrature-point solve that allocates is the usual reason an elimination costs
    # more than the assembly it saves.
    qs, qr = SVector{6}(q), SVector{6}(q_ref)
    for it in 1:(p.local_maxiter)
        r = qs .- qr .- p.Δt .* rate(p.model, ε6, qs)
        if norm(r, Inf) ≤ p.local_tol
            q .= qs
            return (it - 1, true)
        end
        ∂g∂q, = local_tangents(p.model, ε6, qs)
        qs -= (I - p.Δt .* ∂g∂q) \ r
    end
    q .= qs
    return (p.local_maxiter, false)
end

"""
    eliminate!(p, ū; tangents)

Run every quadrature point's local solve at the strains implied by `ū` and store the
resulting stress (and, when asked, the effective tangent).

Threaded over cell chunks, and split the way the local ensemble has to be: the per-point work
writes only to storage its own chunk owns (its `CellValues`, its scratch internal states, its
quadrature points' stresses), while the scatter-add into the global residual and stiffness —
where cells overlap on shared dofs — happens serially in the caller. Regrouping those sums
across threads would change the rounding; this way repeated threaded runs agree exactly.
"""
function eliminate!(p::NortonProblem, ū; tangents::Bool = false, solve::Bool = true)
    ensemble_foreach(p.ensemble, ū, tangents, solve, p) do cells, ichunk, ū, tangents,
            solve, p
        cv, ue = p.cellvalues[ichunk], p.ue[ichunk]
        dofs, coords = p.cdofs[ichunk], p.ccoords[ichunk]
        iters, ok = 0, true
        for cellid in cells
            celldofs!(dofs, p.dh, cellid)
            getcoordinates!(coords, p.dh.grid, cellid)
            Ferrite.reinit!(cv, getcells(p.dh.grid, cellid), coords)
            for a in eachindex(dofs)
                ue[a] = ū[dofs[a]]
            end
            for qp in 1:getnquadpoints(cv)
                idx = qpindex(p, cellid, qp)
                ε6 = tomandel(function_symmetric_gradient(cv, qp, ue))
                if solve
                    # C1: warm-start from the committed state, write only to scratch.
                    q = trial_state(p.buffer, idx)
                    its, converged = solve_qp!(
                        q, p, ε6, view(p.q_ref, (6 * (idx - 1) + 1):(6idx))
                    )
                    iters += its
                    ok &= converged
                else
                    # C3: the committed state already solves the local problems at this `ū`,
                    # so a tangent sweep may read it instead of re-solving.
                    q = committed_state(p.buffer, idx)
                end
                p.σ[idx] = frommandel(SymmetricTensor{2, 3}, stress(p.model, ε6, q))
                if tangents
                    ∂g∂q, ∂g∂ε, ∂σ∂q, ∂σ∂ε = local_tangents(p.model, ε6, q)
                    # Schur corrector of the *time-discrete* local problem, then the
                    # effective consistent tangent that goes into the element stiffness.
                    dqdε = (I - p.Δt .* ∂g∂q) \ (p.Δt .* ∂g∂ε)
                    p.Ceff[idx] = frommandel(SymmetricTensor{4, 3}, ∂σ∂ε + ∂σ∂q * dqdε)
                end
            end
        end
        # One write per chunk rather than one per point: neighbouring reduction slots share a
        # cache line, and updating them inside the point loop makes every chunk invalidate its
        # neighbours' copies on every point.
        p.chunk_iters[ichunk] = iters
        p.chunk_ok[ichunk] = ok
    end
    solve && record_sweep!(p, ū)
    return all(p.chunk_ok)
end

"""
    sweep_is_current(p, ū) / record_sweep!(p, ū)

Whether the last ensemble sweep already solved the local problems at this `ū` *and* at the
tolerance now in force. The tolerance belongs in the key because the solver deliberately
re-commits at a tighter one when it converges — a stale hit there would return the loose
answer as if it were the tight one.
"""
function sweep_is_current(p::NortonProblem, ū)
    p.last_valid[] || return false
    p.last_key[end] == p.local_tol || return false
    @inbounds for i in eachindex(ū)
        p.last_key[i] == ū[i] || return false
    end
    return true
end

function record_sweep!(p::NortonProblem, ū)
    copyto!(view(p.last_key, 1:length(ū)), ū)
    p.last_key[end] = p.local_tol
    p.last_valid[] = true
    return p
end

"Invalidate the sweep key: the committed state or `q_ref` moved under a fixed `ū`."
invalidate_sweep!(p::NortonProblem) = (p.last_valid[] = false; p)

# ── The three multi-level callbacks ─────────────────────────────────────────────────────

"Condensed residual over `ū`: internal force at the eliminated internal state."
function Rbar!(r, ū, p::NortonProblem)
    if !eliminate!(p, ū)
        fill!(r, Inf)          # Inf, never NaN — a line search can back away from Inf
        return nothing
    end
    fill!(r, 0)
    cv, re = p.cellvalues[1], zeros(ndofs_per_cell(p.dh, 1))
    for cellid in 1:getncells(p.dh.grid)
        Ferrite.reinit!(cv, getcells(p.dh.grid, cellid), getcoordinates(p.dh.grid, cellid))
        fill!(re, 0)
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            σ = p.σ[qpindex(p, cellid, qp)]
            for a in 1:length(re)
                re[a] += (shape_symmetric_gradient(cv, qp, a) ⊡ σ) * dΩ
            end
        end
        dofs = celldofs(p.dh, cellid)
        for a in eachindex(dofs)
            r[dofs[a]] += re[a]
        end
    end
    apply_zero!(r, p.ch)       # the correction must not move prescribed dofs
    return nothing
end

"""
    assemble_S!(S, ū, p)

The Schur-condensed tangent, assembled element-wise with a stock Ferrite assembler from the
effective quadrature-point tangent `∂σ/∂ε + ∂σ/∂q · dq/dε`.

Contract C3 guarantees the committed internal state already solves the local problems at this
`ū`, so this is a tangents-only sweep: it reads the committed state rather than re-solving it.
"""
function assemble_S!(S, ū, p::NortonProblem)
    p.nassembly[] += 1
    eliminate!(p, ū; tangents = true, solve = false)
    cv, Ke = p.cellvalues[1], zeros(ndofs_per_cell(p.dh, 1), ndofs_per_cell(p.dh, 1))
    assembler = start_assemble(S)
    for cellid in 1:getncells(p.dh.grid)
        Ferrite.reinit!(cv, getcells(p.dh.grid, cellid), getcoordinates(p.dh.grid, cellid))
        fill!(Ke, 0)
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            Ceff = p.Ceff[qpindex(p, cellid, qp)]
            for a in axes(Ke, 1)
                ∇ˢa_C = shape_symmetric_gradient(cv, qp, a) ⊡ Ceff
                for b in axes(Ke, 2)
                    Ke[a, b] += (∇ˢa_C ⊡ shape_symmetric_gradient(cv, qp, b)) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(p.dh, cellid), Ke)
    end
    # Identity rows and columns on the constrained dofs. The matrix-only `apply!` is the right
    # one here: the residual already went through `apply_zero!`, so there is no right-hand
    # side left to eliminate into.
    apply!(S, p.ch)
    return nothing
end

"""
    commit_internal!(q_dest, ū, p)

Commit step: promote the eliminated internal state at the accepted `ū`.

The solver has just evaluated the residual here, so the scratch state is already the solution
at this `ū` and the commit is a promote. Re-solving would be the third full ensemble sweep at
one iterate — the residual's, this one, and the next tangent assembly's. Idempotent either
way, which is what the commit contract asks for.
"""
function commit_internal!(q_dest, ū, p::NortonProblem)
    ok = sweep_is_current(p, ū) ? all(p.chunk_ok) : eliminate!(p, ū)
    commit_local_state!(p.buffer)
    copyto!(q_dest, vec(p.buffer.committed))
    return ok
end

"""
    monolithic_residual!(F, x, p)

The same time step with nothing eliminated: `x = [ū; vec(Q)]`, the displacement rows carry
equilibrium at the given `Q`, and the internal rows carry the backward-Euler local equation.
Used only as an independent check — no Jacobian, no elimination, no shared tangent code.
"""
function monolithic_residual!(F, x, p::NortonProblem)
    T = eltype(x)
    ū = view(x, 1:(p.n_primary))
    fill!(F, 0)
    cv = p.cellvalues[1]
    ue, re = zeros(T, ndofs_per_cell(p.dh, 1)), zeros(T, ndofs_per_cell(p.dh, 1))
    for cellid in 1:getncells(p.dh.grid)
        Ferrite.reinit!(cv, getcells(p.dh.grid, cellid), getcoordinates(p.dh.grid, cellid))
        dofs = celldofs(p.dh, cellid)
        for a in eachindex(dofs)
            ue[a] = ū[dofs[a]]
        end
        fill!(re, 0)
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            idx = qpindex(p, cellid, qp)
            rng = (6 * (idx - 1) + 1):(6idx)
            ε6 = tomandel(function_symmetric_gradient(cv, qp, ue))
            q = view(x, p.n_primary .+ rng)
            σ = frommandel(SymmetricTensor{2, 3}, stress(p.model, ε6, q))
            for a in eachindex(re)
                re[a] += (shape_symmetric_gradient(cv, qp, a) ⊡ σ) * dΩ
            end
            F[p.n_primary .+ rng] .= q .- view(p.q_ref, rng) .- p.Δt .* rate(p.model, ε6, q)
        end
        for a in eachindex(dofs)
            F[dofs[a]] += re[a]
        end
    end
    # The constrained rows carry `u - u_prescribed` rather than being zeroed. Zeroing is
    # right for a Newton correction (the multi-level path above), but it leaves an all-zero
    # Jacobian row — this form gives the identity row a differentiated solve needs, and it
    # vanishes at any state that satisfies the boundary conditions, so it is equally valid as
    # a residual check.
    for (i, d) in enumerate(p.ch.prescribed_dofs)
        F[d] = x[d] - p.bc_values[i]
    end
    return nothing
end

# ── Assembling the multi-level function ─────────────────────────────────────────────────

function multilevel_function(p::NortonProblem)
    return MultiLevelNonlinearFunction(
        NonlinearFunction(Rbar!; jac = assemble_S!, jac_prototype = copy(p.Sproto));
        primary = 1:(p.n_primary),
        internal = (p.n_primary + 1):(p.n_primary + p.n_internal),
        commit_internal!
    )
end

"Prepare `p` and the full state for the step ending at `t`: freeze `q_ref`, impose the BCs."
function begin_step!(u, p::NortonProblem, t)
    copyto!(p.q_ref, vec(p.buffer.committed))
    invalidate_sweep!(p)          # the local problems changed under an unchanged `ū`
    update!(p.ch, t)
    apply!(view(u, 1:(p.n_primary)), p.ch)
    for (i, d) in enumerate(p.ch.prescribed_dofs)
        p.bc_values[i] = u[d]
    end
    return u
end

# ── The two variants, side by side ──────────────────────────────────────────────────────

"""
    run_variant_b(p, nsteps; warm = true) -> (trajectory, iterations)

Variant B: the condensed problem over `ū`, with the elimination inside the residual.

One cache for the whole time loop. With `warm`, each step is a `reinit!` from the previous
step's solution — the natural time-stepping restart, and what `reinit!` is for. With
`warm = false` the displacement guess is thrown away each step and only the boundary values
are imposed, which is the same sequence of problems from a worse starting point.
"""
function run_variant_b(p::NortonProblem, nsteps; abstol = 1.0e-9, warm::Bool = true)
    u = zeros(p.n_primary + p.n_internal)
    begin_step!(u, p, p.Δt)
    cache = init(
        NonlinearProblem(multilevel_function(p), u, p), MultiLevelNewton();
        abstol, maxiters = 50
    )
    primary = 1:(p.n_primary)
    trajectory, iterations, initial = Vector{Float64}[], Int[], Float64[]
    for step in 1:nsteps
        if step > 1
            u_next = NonlinearSolveBase.get_u(cache)
            warm || fill!(view(u_next, primary), 0)
            begin_step!(u_next, p, step * p.Δt)
            SciMLBase.reinit!(cache, u_next)
        end
        # `reinit!` leaves the residual evaluated at the new starting point, so this is
        # exactly "how good was the guess".
        push!(initial, norm(view(NonlinearSolveBase.get_fu(cache), primary), Inf))
        sol = solve!(cache)
        @assert SciMLBase.successful_retcode(sol) "variant B failed at step $(step)"
        push!(trajectory, copy(sol.u))
        push!(iterations, cache.nsteps)
    end
    return trajectory, iterations, initial
end

"""
    run_variant_a(p, nsteps) -> trajectory

Variant A: the full `[ū; q]` stays the iterate. `CondensedFactorization` keeps the step inside
the displacement block, and `MultiLevelProjection` re-commits the internal variables between
iterations. Plain Newton — this arm supports no globalization.
"""
function run_variant_a(p::NortonProblem, nsteps; abstol = 1.0e-9)
    u = zeros(p.n_primary + p.n_internal)
    mlnf = multilevel_function(p)
    trajectory = Vector{Float64}[]
    for step in 1:nsteps
        begin_step!(u, p, step * p.Δt)
        sol = solve(
            fullspace_problem(mlnf, copy(u), p),
            NewtonRaphson(; linsolve = CondensedFactorization());
            abstol, maxiters = 50, postcondition = MultiLevelProjection(mlnf)
        )
        @assert SciMLBase.successful_retcode(sol) "variant A failed at step $(step)"
        copyto!(u, sol.u)
        push!(trajectory, copy(sol.u))
    end
    return trajectory
end

"Reset the internal state so a variant starts from the same place as the other."
function reset!(p::NortonProblem)
    fill!(p.buffer.committed, 0)
    fill!(p.buffer.scratch, 0)
    fill!(p.q_ref, 0)
    p.nassembly[] = 0
    invalidate_sweep!(p)
    return p
end

# ── Tests ───────────────────────────────────────────────────────────────────────────────

const NSTEPS = 6

@testset "both variants solve the same backward-Euler trajectory" begin
    p = NortonProblem()
    @test p.n_primary == 81           # 27 nodes × 3 displacement components
    @test p.n_internal == 6 * 8 * 8   # 6 Mandel components × 8 cells × 8 quadrature points

    traj_b, iters_b, init_b = run_variant_b(reset!(p), NSTEPS)
    traj_a = run_variant_a(reset!(p), NSTEPS)

    @test length(traj_b) == NSTEPS
    for step in 1:NSTEPS
        ū_b, ū_a = traj_b[step][1:(p.n_primary)], traj_a[step][1:(p.n_primary)]
        q_b, q_a = traj_b[step][(p.n_primary + 1):end], traj_a[step][(p.n_primary + 1):end]
        @test maximum(abs, ū_b .- ū_a) < 1.0e-10
        @test maximum(abs, q_b .- q_a) < 1.0e-10
    end

    # The trajectory is not trivial: the sinusoidal stretch actually moves the solid and the
    # viscous strain actually accumulates.
    @test maximum(abs, traj_b[end][1:(p.n_primary)]) > 1.0e-3
    @test maximum(abs, traj_b[end][(p.n_primary + 1):end]) > 1.0e-4

    # Warm starts. Comparing step 1 against later steps would measure the wrong thing here:
    # the Norton law stiffens as strain accumulates, so the later steps are genuinely harder
    # problems and take *more* iterations even from a good guess. The saving shows up against
    # the same steps solved from a thrown-away guess.
    _, iters_cold, init_cold = run_variant_b(reset!(p), NSTEPS; warm = false)
    @test all(iters_b .≤ iters_cold)
    # Every restarted step starts strictly closer to its solution, and the gap widens as the
    # loading flattens towards the peak of the sinusoid.
    @test all(init_b[2:end] .< init_cold[2:end])
    @test init_b[end] < 0.1 * init_cold[end]
end

@testset "both variants agree with a monolithic solve of the same time step" begin
    # The independent check. The multi-level solves never form the internal equations as
    # residual rows — they eliminate them — so the unreduced problem shares no Jacobian, no
    # condensation and no Schur corrector with them. Here it is solved outright, over the full
    # `[ū; q]`, with a differentiated Jacobian.
    p = NortonProblem()
    traj, = run_variant_b(reset!(p), NSTEPS)

    # `p` is left holding `q_ref` and the boundary values of the final step, so the monolithic
    # residual below is that step's equations.
    F = zeros(p.n_primary + p.n_internal)
    monolithic_residual!(F, traj[end], p)
    @test maximum(abs, view(F, 1:(p.n_primary))) < 1.0e-9
    @test maximum(abs, view(F, (p.n_primary + 1):length(F))) < 1.0e-10

    reference = solve(
        NonlinearProblem(NonlinearFunction(monolithic_residual!), copy(traj[end - 1]), p),
        NewtonRaphson(; autodiff = AutoForwardDiff()); abstol = 1.0e-11, maxiters = 30
    )
    @test SciMLBase.successful_retcode(reference)
    @test maximum(abs, reference.u .- traj[end]) < 1.0e-10
end

@testset "the global Newton converges quadratically within a step" begin
    p = NortonProblem()
    reset!(p)
    u = zeros(p.n_primary + p.n_internal)
    begin_step!(u, p, p.Δt)

    cache = init(
        NonlinearProblem(multilevel_function(p), u, p),
        MultiLevelNewton(; jacobian_reuse = :always); abstol = 1.0e-12, maxiters = 50
    )
    residuals = residual_history(cache; maxsteps = 20)
    @test SciMLBase.successful_retcode(cache.retcode)
    @test length(residuals) ≥ 4                      # enough for an order estimate at all
    @test tail_order(residuals; floor = 1.0e-13) > 1.8

    # One assembly per iteration under `:always`, and nothing assembles behind the cache's back.
    @test p.nassembly[] == cache.stats.njacs
end

@testset "threaded elimination is reproducible" begin
    runs = map(1:2) do _
        p = NortonProblem(; nchunks = 4)
        first(run_variant_b(reset!(p), 3))
    end
    @test runs[1] == runs[2]

    serial = first(run_variant_b(reset!(NortonProblem(; nchunks = 1)), 3))
    @test maximum(abs, runs[1][end] .- serial[end]) < 1.0e-12
end
