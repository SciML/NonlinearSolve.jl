# [Multi-Level Newton for Problems with Internal Variables](@id multilevel_newton)

Some nonlinear systems have unknowns that split into two very differently shaped blocks. A
finite-element solid with internal variables is the standard example: the nodal
displacements `ū` are globally coupled, while the internal variables `q` (plastic strain,
damage, viscoelastic state) satisfy their own small nonlinear system at each quadrature
point and are coupled to nothing but the strain there.

Solving `[ū; q]` monolithically wastes that structure. **Multi-level Newton**
(Rabbat–Sangiovanni-Vincentelli–Hsieh) instead eliminates `q` at fixed `ū` — one small
independent solve per point — and takes a Newton step on the Schur-condensed system

```math
S \, \delta\bar{u} = -\bar{R}, \qquad
S = \frac{\partial \bar{R}}{\partial \bar{u}}
  + \frac{\partial \bar{R}}{\partial q}\frac{\mathrm{d}q}{\mathrm{d}\bar{u}}.
```

The cross blocks are never formed. You assemble `S` yourself from per-point correctors
`dq/dε`, exactly the way an element tangent is assembled, and the solver never sees
`J_ūq`, `J_qq` or `J_qū`.

## The three callbacks

A [`MultiLevelNonlinearFunction`](@ref) is built from a condensed `NonlinearFunction` over
`ū` plus a commit step:

| callback | signature | role |
|:---|:---|:---|
| `Rbar!` | `(r, ū, p)` | run the local ensemble at `ū`, write the condensed residual |
| `assemble_S!` | `(S, ū, p)` | assemble the Schur tangent from per-point correctors |
| `commit_internal!` | `(q_dest, ū, p) -> Bool` | promote the local state at an accepted `ū` |

`primary` and `internal` say where `ū` and `q` live in the full state, so `sol.u` comes back
as the whole `[ū; q]` and can be compared directly against a monolithic solve.

## The trial/commit contract

The solver calls `Rbar!` at points it has not accepted — line-search trial points, trust
region proposals — so the three callbacks are bound by a contract:

  - **C1 (trials).** Every `Rbar!` and `assemble_S!` call is a trial. It must not mutate the
    committed internal state, and its local solves should warm-start from it. This is what
    makes a line search's `ϕ(α)` a function of `α` alone rather than of the order the trials
    happened in. It is also a correctness requirement, not just an efficiency one, when a
    local problem has several roots (return-mapping plasticity, damage): the warm start is
    what selects the branch.

  - **C2 (commit).** `commit_internal!` runs exactly once per accepted global iterate. It
    re-solves the locals there, promotes them to committed, writes them into `q_dest`, and
    reports whether they all converged.

  - **C3 (assembly).** `assemble_S!` is only ever called at an iterate whose committed state
    is already consistent with it, under every globalization. It may therefore read the
    committed tangents instead of re-solving — that is the whole point of the elimination.

[`LocalStateBuffer`](@ref) is the double buffer C1 asks for, and
[`ensemble_foreach`](@ref) runs the points in chunks, in parallel, without ever indexing by
`threadid()` (a task can migrate mid-run, so chunk index is the only valid key for per-worker
storage).

## Four independent knobs

Accuracy and preconditioning enter at four places that are easy to confuse and are never
conflated in this API. `MultiLevelNewton` itself exposes none of them.

| knob | what it controls | where it is set |
|:---|:---|:---|
| local forcing | how accurately `q` is eliminated | [`LocalToleranceSchedule`](@ref) on the function |
| linear forcing | how accurately `S·δū = -R̄` is solved | `forcing` on the `global_solver` |
| linear preconditioning | the preconditioner for that linear solve | `precs` on the `global_solver`'s `linsolve` |
| nonlinear preconditioning | a corrector applied to the iterate | the `postcondition` solve keyword |

The **local forcing** knob is the one specific to this solver. Solving every local problem
to full precision while the global iterate is still far from the root is wasted work, and
perturbed-Newton theory says the global rate survives as long as the local accuracy shrinks
at least as fast as the rate being claimed — `:quadratic` pairs with
`jacobian_reuse = :always`, `:linear` with `:chord`. The schedule's floor bounds how accurate
the committed `q` can ever be, so the residual of the *unreduced* problem at the returned
root is only good to about `abstol + L_q · floor`. Leave `local_tolerance` at `nothing` when
you need the tight bound.

Read the current tolerance with [`local_tolerance`](@ref) and unwrap your own parameters with
[`user_parameters`](@ref); both work whether or not a schedule is configured, so the same
callbacks run unchanged either way.

## What failure looks like

| what went wrong | how to report it | what you see |
|:---|:---|:---|
| a local solve diverged at a **trial** | write `Inf` into the affected residual rows | a backtracking line search halves the step and recovers |
| a local solve diverged at a **commit** | return `false` from `commit_internal!` | `ReturnCode.ConvergenceFailure` |
| the same with no globalization | the `Inf` reaches the convergence test | `ReturnCode.Unstable` |
| the condensed linear solve failed | nothing to do | `ReturnCode.InternalLinearSolveFailed` |

Write `Inf`, never `NaN`. A backtracking line search decides to shrink the step by comparing
merit values; `NaN` fails that comparison in both directions, so the search evaluates the
residual at `NaN` states until it runs out of iterations instead of backing off.

Globalization for v1 is `BackTracking` or none. `LiFukushima` aborts on a non-finite initial
merit and `RobustNonMonotone` has no finiteness guard at all, so neither can recover from a
diverged trial.

## A worked example

A 1-D bar of linear elements, one quadrature point each, with three internal variables per
element obeying `q = γ·tanh(c·ε + D·q)` and a stress `σ = E(ε - m·q)`. Equilibrium says the
stress is the same in every element and equal to the applied load.

```@example multilevel
using NonlinearSolve, LinearAlgebra, SparseArrays

const γ, E, f_ext = 0.7, 1.0, 0.3
const c = [2.0, -3.0, 2.5]
const m = [0.4, 0.3, 0.2]
const D = [0.25 0.10 -0.15; 0.10 0.20 0.10; -0.15 0.10 0.30]

struct Bar
    n::Int
    h::Float64
    buffer::LocalStateBuffer{Matrix{Float64}}
    ensemble::LocalEnsemble
    σ::Vector{Float64}
    Dt::Vector{Float64}
    ok::Vector{Bool}
end

function Bar(n; nchunks = Threads.nthreads())
    ens = LocalEnsemble(n; nchunks)
    Bar(n, 1 / n, LocalStateBuffer(zeros(3, n)), ens,
        zeros(n), zeros(n), fill(true, length(ens.chunks)))
end

strain(ū, bar, i) = (ū[i] - (i == 1 ? zero(eltype(ū)) : ū[i - 1])) / bar.h
nothing # hide
```

The local problem and its analytic corrector. `dq/dε = (I - γ ∂g/∂q) \ (γ ∂g/∂ε)` follows
from differentiating the local residual at fixed `ε`, and `dσ/dε = E(1 - m·dq/dε)` is the
element tangent that goes into `S`.

```@example multilevel
function local_solve!(q, ε, tol)
    for _ in 1:50
        g = tanh.(c .* ε .+ D * q)
        r = q .- γ .* g
        norm(r, Inf) ≤ tol && return true
        q .-= (I - γ .* ((1 .- g .^ 2) .* D)) \ r
    end
    return false
end

function element_tangent(ε, q)
    g = tanh.(c .* ε .+ D * q)
    s = 1 .- g .^ 2
    dqdε = (I - γ .* (s .* D)) \ (γ .* (s .* c))
    return E * (1 - dot(m, dqdε))
end
nothing # hide
```

The ensemble runs in chunks. Each chunk writes only its own points and its own reduction
slot; the reduction itself is folded serially afterwards, so a threaded run reproduces
itself exactly.

```@example multilevel
function run_ensemble!(bar, ū, tol)
    fill!(bar.ok, true)
    ensemble_foreach(bar.ensemble, ū, tol, bar) do chunk, ichunk, ū, tol, bar
        for i in chunk
            ε = strain(ū, bar, i)
            q = trial_state(bar.buffer, i)          # warm-started from committed state
            local_solve!(q, ε, tol) || (bar.ok[ichunk] = false)
            bar.σ[i] = E * (ε - dot(m, q))
        end
    end
    return all(bar.ok)
end

function Rbar!(r, ū, p)
    bar = user_parameters(p)
    tol = something(local_tolerance(p), 1.0e-14)
    run_ensemble!(bar, ū, tol) || (fill!(r, Inf); return nothing)   # Inf, never NaN
    for j in 1:(bar.n - 1)
        r[j] = bar.σ[j] - bar.σ[j + 1]
    end
    r[bar.n] = bar.σ[bar.n] - f_ext
    return nothing
end
nothing # hide
```

`assemble_S!` runs no local solves — C3 guarantees the committed state is already the root at
this `ū`. With `∂ε_i/∂u_i = 1/h` and `∂ε_i/∂u_{i-1} = -1/h`, the rows `R̄_j = σ_j - σ_{j+1}`
give a symmetric tridiagonal.

```@example multilevel
function assemble_S!(S, ū, p)
    bar = user_parameters(p)
    for i in 1:(bar.n)
        bar.Dt[i] = element_tangent(strain(ū, bar, i), committed_state(bar.buffer, i))
    end
    fill!(S, 0)
    for j in 1:(bar.n - 1)
        S[j, j] = (bar.Dt[j] + bar.Dt[j + 1]) / bar.h
        S[j, j + 1] = S[j + 1, j] = -bar.Dt[j + 1] / bar.h
    end
    S[bar.n, bar.n] = bar.Dt[bar.n] / bar.h
    return nothing
end

function commit_internal!(q_dest, ū, p)
    bar = user_parameters(p)
    ok = run_ensemble!(bar, ū, something(local_tolerance(p), 1.0e-14))
    commit_local_state!(bar.buffer)
    copyto!(q_dest, vec(bar.buffer.committed))
    return ok
end
nothing # hide
```

Putting it together:

```@example multilevel
n = 40
bar = Bar(n)
S = spdiagm(-1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1))

f = MultiLevelNonlinearFunction(
    NonlinearFunction(Rbar!; jac = assemble_S!, jac_prototype = S);
    primary = 1:n, internal = (n + 1):(4n), commit_internal!
)

sol = solve(NonlinearProblem(f, zeros(4n), bar), MultiLevelNewton(); abstol = 1.0e-12)
sol.retcode, sol.u[n], sol.stats
```

`sol.u` is the full `[ū; q]`: the first `n` entries are the displacements, the rest the
committed internal variables. The residual rows belonging to `q` are structurally zero,
because those equations were eliminated rather than solved.

### Watching it converge

Driving the cache one step at a time shows the quadratic rate, and shows what freezing the
tangent costs:

```@example multilevel
function history(cache; chord = false)
    e = [norm(view(NonlinearSolveBase.get_fu(cache), 1:n), Inf)]
    k = 0
    while NonlinearSolveBase.not_terminated(cache) && k < 100
        k += 1
        chord ? step!(cache; recompute_jacobian = k <= 1) : step!(cache)
        push!(e, norm(view(NonlinearSolveBase.get_fu(cache), 1:n), Inf))
    end
    return e
end

prob = NonlinearProblem(f, zeros(4n), Bar(n))
history(init(prob, MultiLevelNewton(); abstol = 1.0e-12))
```

```@example multilevel
prob_chord = NonlinearProblem(f, zeros(4n), Bar(n))
history(
    init(prob_chord, MultiLevelNewton(; jacobian_reuse = :chord); abstol = 1.0e-12);
    chord = true
)
```

The first sequence squares its error each step; the second reduces it by a constant factor,
which is what a frozen tangent buys — one assembly and one factorization for the whole solve.

## The full-space arm

Everything above eliminates `q` *inside* the residual and hands the framework a problem over
`ū` alone. There is a second way to arrange the same mathematics: keep the full `[ū; q]` as
the iterate, let a δq-zeroing linear solver confine the step to the primary block, and run the
commit as an iterate corrector between iterations. Same Schur tangent, same root — different
plumbing.

```@example multilevel
mlnf = MultiLevelNonlinearFunction(
    NonlinearFunction(Rbar!; jac = assemble_S!, jac_prototype = S);
    primary = 1:n, internal = (n + 1):(4n), commit_internal!
)

full = fullspace_problem(mlnf, zeros(4n), Bar(n))
sol_a = solve(
    full, NewtonRaphson(; linsolve = CondensedFactorization());
    abstol = 1.0e-12, postcondition = MultiLevelProjection(mlnf)
)
sol_a.retcode, sol_a.u[n]
```

Three pieces, and they only work together:

  - [`fullspace_problem`](@ref) builds a plain `NonlinearFunction` whose `jac_prototype` is a
    [`SchurOperator`](@ref) — formally `n × n`, storing only the `n̄ × n̄` block, because the
    cross blocks are never formed.
  - [`CondensedFactorization`](@ref) solves `S·δū = -R̄` with its `inner` algorithm and zeros
    the internal block of the step. `inner` is the top-level algorithm of a real nested
    `LinearSolve` cache, which is what makes its `precs` take effect at all.
  - [`MultiLevelProjection`](@ref) runs `commit_internal!` on the internal block at every
    accepted iterate, so `q` solves its local problems at the `ū` just accepted.

### Which arm to use

Prefer [`MultiLevelNewton`](@ref) unless you have a specific reason not to. It globalizes
correctly, it carries the local-forcing schedule, and it is the arm the failure semantics
above describe.

The full-space arm exists for the cases where the framework has to see `q` in the iterate —
an integrator applying its own error control to the internal variables, or a continuation
scheme that wants the whole state — and for solvers reached only through the `postcondition`
option. Its cost is that **it supports no globalization**: plain Newton only.

It also has no failure channel of its own: the internal residual rows are zero by
construction, so a commit that fails is not observed where it happens. It surfaces one
iteration later as an `Inf` in the *primary* rows, and there is no `ConvergenceFailure`
equivalent — that belongs to [`MultiLevelNewton`](@ref), which owns the commit.

The globalization restriction is not a missing feature, it is what the arrangement implies. A line search
scores the step on the residual *before* the corrector runs, and at that point the internal
rows are `R_q(ū + αδū, q_prev)`, which grow with `α` — so the Armijo condition can fail on a
perfectly well-posed problem. A trust region is worse: its Dogleg builds the Cauchy leg from
steepest descent, which never passes through the linear solver, so the internal block of the
step stops being zero at all. Both pairings are refused with an error naming the cause rather
than being allowed to mis-solve, and `concrete_jac = true` is refused too, since the operator
stands for a matrix that does not exist.

Local forcing is also unavailable on this arm: the tolerance cell belongs to a multi-level
cache, and there is none here, so [`local_tolerance`](@ref) returns `nothing` and your
callbacks run at their own fixed tolerance.

## Choosing a global solver

```julia
MultiLevelNewton(;
    global_solver = NewtonRaphson(;
        linsolve = KrylovJL_GMRES(; precs = my_precs), concrete_jac = true
    ),
    jacobian_reuse = :chord
)
```

Any algorithm with a stepping cache works — `NewtonRaphson`, `TrustRegion`. Two constraints:

  - Pass `concrete_jac = true` with a Krylov linear solver. Without it the global solver
    treats the problem as matrix-free and never calls `assemble_S!` at all, and a multi-level
    problem has no matrix-free Jacobian to fall back on: `S` is assembled, not
    differentiated.
  - `lb`/`ub` bounds and the `precondition` keyword are rejected. Both would be composed into
    the condensed function, where they are the wrong length — the corrector acts on the full
    residual while the solved system is the condensed one. Impose them inside your own
    residual instead.

`precs` is rebuilt exactly when `S` is assembled, so under `jacobian_reuse = :chord` an
expensive preconditioner setup is paid once rather than every step.
