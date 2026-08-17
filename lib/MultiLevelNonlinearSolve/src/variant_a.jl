#
# The full-space arm. Instead of driving a condensed subcache, the solver iterates on the
# whole `[ū; q]`: a δq-zeroing linear solver keeps the step inside the primary block, and the
# commit runs as a `postcondition` corrector between iterations. Same root, same Schur
# tangent, different plumbing — see the tutorial for which arm to choose.
#
# Scope: `:None` globalization only. A line search's merit and a trust region's ratio are
# computed on the *unprojected* full residual (the corrector runs after globalization), where
# the internal rows grow with the step length, so Armijo can reject steps on a well-posed
# problem; and a trust region's Dogleg builds its Cauchy leg from steepest descent, which
# never passes through the linear solver and so breaks `δu[q] = 0` outright.
#

"""
    SchurOperator(S, primary, internal)

The condensed Schur tangent `S` presented at full size. It reports `n × n` where `n` is the
length of the full state, but stores only the `n̄ × n̄` block — the point of the elimination is
that the cross blocks are never formed, so there is nothing else to store.

Used as the `jac_prototype` of the full-space problem, where it is handed to the linear
solver untouched rather than being differentiated or copied. Its matrix product is the
embedded condensed action — `S` on the primary rows, zero on the internal rows — which is the
step [`CondensedFactorization`](@ref) produces; it is not the Jacobian of the full residual,
which is exactly why this arm needs both pieces together.
"""
struct SchurOperator{T, M, P, I} <: SciMLOperators.AbstractSciMLOperator{T}
    S::M
    primary::P
    internal::I
    n::Int
end

function SchurOperator(S::AbstractMatrix, primary, internal)
    return SchurOperator{eltype(S), typeof(S), typeof(primary), typeof(internal)}(
        S, primary, internal, length(primary) + length(internal)
    )
end

Base.size(op::SchurOperator) = (op.n, op.n)
SciMLOperators.isconstant(::SchurOperator) = false

function LinearAlgebra.mul!(v::AbstractVector, op::SchurOperator, u::AbstractVector)
    LinearAlgebra.mul!(view(v, op.primary), op.S, view(u, op.primary))
    fill!(view(v, op.internal), zero(eltype(v)))
    return v
end

function LinearAlgebra.mul!(
        v::AbstractVector, op::SchurOperator, u::AbstractVector, α::Number, β::Number
    )
    LinearAlgebra.mul!(view(v, op.primary), op.S, view(u, op.primary), α, β)
    rmul!(view(v, op.internal), β)
    return v
end

Base.:*(op::SchurOperator, u::AbstractVector) = LinearAlgebra.mul!(similar(u), op, u)

# It genuinely cannot be materialised: the full-size matrix it stands for does not exist,
# only its primary block does. Declaring that keeps the guards that check the trait on the
# right branch, and the `convert` method turns the paths that ignore the trait — a trust
# region's steepest-descent leg is the one that matters — into an explanation rather than a
# `MethodError` about an operator the user never constructed.
SciMLOperators.isconvertible(::SchurOperator) = false

# The three sites below are backstops for the same scope rule, each covering a different way
# an unsupported configuration reaches the code first:
#   * `convert`         — a trust region, whose steepest-descent leg wants a concrete matrix;
#   * `SchurAssembly`   — a line search, whose slope is formed outside the Jacobian cache;
#   * `check_none_globalization` — anything that reaches the corrector with a globalization
#     still set, which the two above normally intercept first.
function _globalization_unsupported(site::AbstractString)
    return ArgumentError(
        site * " The full-space multi-level arm supports no globalization: a line search \
         scores the step on the unprojected residual, whose internal rows grow with the step \
         length, and a trust region's Dogleg builds its Cauchy leg outside the linear solver, \
         so the internal block of the step would stop being zero. Use a plain \
         `NewtonRaphson()` here, or switch to `MultiLevelNewton`, which globalizes on the \
         condensed problem."
    )
end

function Base.convert(::Type{AbstractMatrix}, ::SchurOperator)
    throw(
        _globalization_unsupported(
            "a `SchurOperator` cannot be converted to a matrix: it stands for an `n × n` \
             Jacobian that is never formed, and only its `n̄ × n̄` primary block exists."
        )
    )
end

"""
    CondensedFactorization(inner = LUFactorization())

Linear solver for the full-space arm: solves `S·δū = -R̄` on the primary block with `inner`
and zeros the internal block of the solution, so the step never moves `q` — the projection
owns that.

`inner` is the top-level algorithm of a genuine nested `LinearSolve` cache, which is what
makes its `precs` take effect: `precs` is read only from the algorithm handed to `init`, so a
wrapper that merely forwards `solve!` would degrade preconditioning to the identity without
saying so.

Being an `AbstractFactorization` is also load-bearing. It is what makes the Jacobian-reuse
signal reach the linear solver: on a reused Jacobian the cache is not marked stale, so
neither the inner factorization nor the preconditioner is rebuilt.
"""
@concrete struct CondensedFactorization <: LinearSolve.AbstractFactorization
    inner
end

CondensedFactorization(; inner = LinearSolve.LUFactorization()) =
    CondensedFactorization(inner)

# The operator is never converted to a matrix: only its stored `S` block is factored, and
# that is reached through the cacheval below. Without this the `isconvertible` guard in the
# Jacobian cache would assert a contract that is then quietly unmet.
LinearSolve.needs_concrete_A(::CondensedFactorization) = false
# Chosen deliberately rather than inherited: `set_lincache_A!` re-queries these on every
# Jacobian update, and the default `false` would make it `copyto!` into the operator.
LinearSolve.default_alias_A(::CondensedFactorization, ::Any, ::Any) = true
LinearSolve.default_alias_b(::CondensedFactorization, ::Any, ::Any) = false

# `verbose` and `assumptions` are typed exactly as in LinearSolve's generic
# `AbstractFactorization` method: leaving them untyped makes this method more specific in the
# leading arguments and less specific in the trailing ones, which is an ambiguity rather than
# an override.
function LinearSolve.init_cacheval(
        alg::CondensedFactorization, A::SchurOperator, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Union{LinearSolve.LinearVerbosity, Bool},
        assumptions::LinearSolve.OperatorAssumptions
    )
    n̄ = length(A.primary)
    # Never factor the live `S`: a destructive backend (dense `lu!`) would overwrite the
    # matrix `assemble_S!` writes into. One copy per refactorization is nothing next to the
    # assembly sweep that produced it.
    S_work = copy(A.S)
    b̄ = similar(b, n̄)
    ū = similar(u, n̄)
    fill!(b̄, zero(eltype(b̄)))
    fill!(ū, zero(eltype(ū)))
    inner_cache = LinearSolve.init(
        LinearSolve.LinearProblem(S_work, b̄; u0 = ū), alg.inner;
        alias = LinearSolve.LinearAliasSpecifier(alias_A = true, alias_b = true),
        abstol, reltol, maxiters
    )
    return (; inner_cache, S_work, b̄)
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::CondensedFactorization; kwargs...
    )
    A = cache.A::SchurOperator
    (; inner_cache, S_work, b̄) = cache.cacheval

    if cache.isfresh
        # Propagate freshness: snapshot the live `S` and reassign the inner `A`, which is
        # what marks the inner cache for refactorization and its preconditioner for rebuild.
        copyto!(S_work, A.S)
        inner_cache.A = S_work
        cache.isfresh = false
    end

    copyto!(b̄, view(cache.b, A.primary))
    inner_cache.b = b̄
    inner_sol = SciMLBase.solve!(inner_cache; kwargs...)

    copyto!(view(cache.u, A.primary), inner_sol.u)
    fill!(view(cache.u, A.internal), zero(eltype(cache.u)))

    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = inner_sol.retcode
    )
end

# Eisenstat–Walker forcing calls `update_tolerances!` unconditionally, and the default hook
# throws for an algorithm that never defines it. Delegate to the cache that actually holds
# the tolerances — the inner one.
function LinearSolve.update_tolerances_internal!(
        cache, ::CondensedFactorization, abstol, reltol
    )
    LinearSolve.update_tolerances!(cache.cacheval.inner_cache; abstol, reltol)
    return nothing
end

"""
    MultiLevelProjection(mlnf)

The `postcondition` corrector for the full-space arm: at every accepted iterate it runs
`commit_internal!` on the internal block, so `q` is the solution of the local problems at the
`ū` the solver just accepted. On this arm the commit *is* the projection.

It is idempotent — committing twice at the same `ū` gives the same `q` — and the trial
contract C1 is satisfied for free, since the step never moves `q` for a trial to disturb.

A commit that reports failure has no direct channel on this arm: the internal residual rows
are zero by construction, so nothing observes the flag at the point it is raised. It surfaces
one iteration later, when the next trial evaluates the condensed residual at an internal state
that does not solve its local problems and writes `Inf` into the *primary* rows. There is no
`ConvergenceFailure` equivalent here — that is `MultiLevelNewton`'s, which owns the commit.
"""
struct MultiLevelProjection{P, I, C}
    primary::P
    internal::I
    commit_internal!::C
end

MultiLevelProjection(mlnf::MultiLevelNonlinearFunction) =
    MultiLevelProjection(mlnf.primary, mlnf.internal, mlnf.commit_internal!)

function (projection::MultiLevelProjection)(u, u_prev, p, cache)
    cache === nothing || check_none_globalization(cache)
    projection.commit_internal!(
        view(u, projection.internal), view(u, projection.primary), p
    )
    return u
end

function check_none_globalization(cache)
    globalization = Utils.safe_getproperty(cache, Val(:globalization))
    (globalization === missing || globalization isa Val{:None}) && return nothing
    throw(
        _globalization_unsupported(
            "the projection was reached with the globalization $(globalization) in effect."
        )
    )
end

# `fullspace_problem` builds a *plain* `NonlinearFunction` rather than a
# `MultiLevelNonlinearFunction`: the Jacobian path is typed to `NonlinearFunction`, so a
# wrapper type never reaches it. The wrapper is still the residual — it is already callable at
# full length — it just travels as the plain function's callable rather than as its type.
struct SchurAssembly{J, P}
    jac::J
    primary::P
end

function (assembly::SchurAssembly)(op::SchurOperator, u, p)
    assembly.jac(op.S, view(u, assembly.primary), p)
    return nothing
end

# The only thing that asks for the Schur tangent in storage other than the operator is a line
# search's directional derivative, which falls back to a freshly allocated dense `n × n`
# matrix when the function has an analytic `jac` and no `jvp`. On this arm that is a
# configuration error rather than a performance one, so say which.
function (::SchurAssembly)(J, u, p)
    throw(
        _globalization_unsupported(
            "the Schur tangent was requested in a dense `$(nameof(typeof(J)))` rather than \
             the `SchurOperator` it is assembled into."
        )
    )
end

"""
    fullspace_problem(mlnf, u0, p = SciMLBase.NullParameters())

Build the full-space `NonlinearProblem` for the same callbacks a [`MultiLevelNewton`](@ref)
solve would use. Solve it with a plain Newton, the δq-zeroing linear solver and the
projection:

```julia
prob = fullspace_problem(mlnf, u0, p)
sol = solve(
    prob, NewtonRaphson(; linsolve = CondensedFactorization()),
    postcondition = MultiLevelProjection(mlnf)
)
```

The residual is the wrapper's full-space form — the condensed residual in the primary rows,
zeros in the internal rows, since the internal equations are satisfied by the projection
rather than by the linear solve.

Local forcing is not available on this arm: there is no multi-level cache to own the
tolerance cell, so the callbacks see the parameters unchanged and
[`local_tolerance`](@ref) returns `nothing`.
"""
function fullspace_problem(
        mlnf::MultiLevelNonlinearFunction, u0, p = SciMLBase.NullParameters()
    )
    SciMLBase.has_jac(mlnf.f) || throw(
        ArgumentError(
            "the full-space arm needs the analytic Schur tangent: give the condensed \
             function a `jac`. It has no way to differentiate the elimination."
        )
    )
    mlnf.f.jac_prototype isa AbstractMatrix || throw(
        ArgumentError(
            "the full-space arm needs a matrix `jac_prototype` on the condensed function to \
             store `S` in; got a `$(typeof(mlnf.f.jac_prototype))`."
        )
    )
    f = SciMLBase.NonlinearFunction{true}(
        mlnf;
        jac = SchurAssembly(mlnf.f.jac, mlnf.primary),
        jac_prototype = SchurOperator(
            copy(mlnf.f.jac_prototype), mlnf.primary, mlnf.internal
        )
    )
    return SciMLBase.NonlinearProblem(f, u0, p)
end
