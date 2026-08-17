"""
    MultiLevelNonlinearFunction(f; primary, internal, commit_internal!,
                                local_tolerance = nothing)

A nonlinear function whose unknowns split into a *primary* block `ū` and an *internal* block
`q` that decouples into one small independent problem per point, so `q` can be eliminated at
fixed `ū`. [`MultiLevelNewton`](@ref) solves the resulting condensed problem over `ū` alone.

### Arguments

  - `f`: the condensed `NonlinearFunction` over `ū`, in place.

      + `f.f` is `Rbar!(r, ū, p)`: run the local ensemble at `ū` and write the condensed
        residual. Every call is a *trial* — it must not mutate the committed internal state,
        and it should warm-start the local solves from it. Write `Inf` (never `NaN`) into
        the residual rows a diverged local solve makes meaningless.
      + `f.jac` is `assemble_S!(S, ū, p)`: the Schur tangent
        `S = ∂R̄/∂ū + (∂R̄/∂q)(dq/dū)`, assembled from per-point correctors. It is only ever
        called at an iterate whose committed internal state is already consistent with `ū`,
        so it can read the committed tangents instead of re-solving.
      + `f.jac_prototype`: the storage for `S`.

  - `primary`: indices of `ū` in the full state (a `UnitRange` is the fast path).
  - `internal`: indices of `q` in the full state.
  - `commit_internal!`: `(q_dest, ū, p) -> Bool`. Promotes the local state at the accepted
    `ū` to committed, writes it into `q_dest`, and reports whether every local solve
    converged.

    It must be **idempotent at fixed `ū`**: re-solving from the committed state and promoting
    again has to leave the same `q`. The solver calls it more than once at the same iterate —
    at `init`, after a corrector moves `ū`, when tightening the elimination at convergence,
    and when restoring a best iterate — so a commit that *appends* to a history rather than
    recomputing it will drift.
  - `local_tolerance`: a [`LocalToleranceSchedule`](@ref) or `nothing`.

The function is callable at *full* length: it writes the condensed residual into the primary
rows and zeros the internal rows, so `sol.u` is the full `[ū; q]` and `sol.resid` can be
compared against a monolithic solve of the same problem.
"""
struct MultiLevelNonlinearFunction{iip, F, P, I, C, T} <:
       SciMLBase.AbstractNonlinearFunction{iip}
    f::F
    primary::P
    internal::I
    commit_internal!::C
    local_tolerance::T
end
# `remake` and the initialization machinery reach for a couple of `NonlinearFunction` fields.
# They are answered here rather than forwarded to the condensed function, whose versions are
# stated in `ū` coordinates while this function is full length; v1 supports neither symbolic
# indexing nor initialization data on a multi-level problem.
#
# Nothing else is forwarded, and there are deliberately no `has_jac`/`has_jvp` methods: the
# condensed `jac` is `n̄ × n̄` while this residual has length `n`, so a consumer that trusted a
# forwarded `has_jac` would pair the two and silently mis-size the linear solve. Only
# `MultiLevelNewton`'s `__init` reads the `f` field, and it does so explicitly.
function Base.getproperty(mlnf::MultiLevelNonlinearFunction, name::Symbol)
    (name === :sys || name === :initialization_data) && return nothing
    return getfield(mlnf, name)
end

# The multi-level structure has nothing for `remake` to rebuild; it is only reached because
# `remake` on the problem re-derives the function object.
function SciMLBase.remake(
        mlnf::MultiLevelNonlinearFunction; f = missing, initialization_data = missing,
        kwargs...
    )
    (f === missing || f === mlnf) || throw(
        ArgumentError("cannot `remake` the residual of a `MultiLevelNonlinearFunction`.")
    )
    (initialization_data === missing || initialization_data === nothing) || throw(
        ArgumentError(
            "`MultiLevelNonlinearFunction` does not support `initialization_data`."
        )
    )
    return mlnf
end

function MultiLevelNonlinearFunction(
        f::SciMLBase.NonlinearFunction{iip}; primary, internal, commit_internal!,
        local_tolerance = nothing
    ) where {iip}
    iip || throw(
        ArgumentError(
            "`MultiLevelNonlinearFunction` requires an in-place condensed function: the \
             local ensemble is eliminated through mutating workspaces."
        )
    )
    return MultiLevelNonlinearFunction{
        iip, typeof(f), typeof(primary), typeof(internal),
        typeof(commit_internal!), typeof(local_tolerance),
    }(f, primary, internal, commit_internal!, local_tolerance)
end

function MultiLevelNonlinearFunction(f; kwargs...)
    return MultiLevelNonlinearFunction(SciMLBase.NonlinearFunction{true}(f); kwargs...)
end

# Composing a left preconditioner is refused here rather than at `__init`, because the
# conditioning pass runs first and would otherwise fail while rebuilding the wrapper, with an
# error about an internal type the user never named.
function NonlinearSolveBase.compose_precondition(
        ::MultiLevelNonlinearFunction, pre, ::Val
    )
    throw(
        ArgumentError(
            "`precondition` is not supported on a multi-level problem. The corrector `G` \
             acts on the full residual, of length `n`, while the system actually solved is \
             the condensed one of length `n̄` — composing `G` onto it is not just a different \
             preconditioner, it is the wrong size. Compose `G` into your own condensed \
             residual instead."
        )
    )
end

function (mlnf::MultiLevelNonlinearFunction{true})(res, u, p)
    mlnf.f(view(res, mlnf.primary), view(u, mlnf.primary), p)
    fill!(view(res, mlnf.internal), zero(eltype(res)))
    return nothing
end

# The two halves of the auto-wired analytic `jvp`, installed by `__init` when the user
# supplies none. Without a `jvp`, a line search's directional derivative takes the fallback
# at SciMLJacobianOperators.jl:378-388, which allocates a *dense* `n̄ × n̄` matrix and calls
# `assemble_S!` into it once per step — even when `α = 1` is accepted immediately, since
# `ϕ'(0)` is always formed. At FEM sizes that allocation alone is fatal, and a sparse
# assembler would not even accept the dense destination. `last_S` records whatever storage
# the framework hands the assembler, so the slope is taken at the last assembled `S`: the
# chord slope, which is what `jacobian_reuse` promises in any case.
struct TrackedSchurAssembly{J, R}
    jac::J
    last_S::R
end

function (assembly::TrackedSchurAssembly)(S, ū, p)
    assembly.jac(S, ū, p)
    assembly.last_S[] = S
    return nothing
end

struct SchurJacVecProduct{R}
    last_S::R
end

function (jvp::SchurJacVecProduct)(Jv, v, ū, p)
    S = jvp.last_S[]
    # Reached before any assembly. That means the global solver never builds a concrete
    # Jacobian at all — a Krylov linear solver treats the problem as matrix-free and drives
    # the solve through this product — so there is no Schur matrix to multiply by.
    S === nothing && throw(
        ArgumentError(
            "the Schur tangent has not been assembled, so it cannot be applied. The global \
             solver is running matrix-free, which a multi-level problem cannot do: `S` is \
             assembled by `jac`, not differentiated. Pass `concrete_jac = true` to the \
             global solver (`NewtonRaphson(; linsolve = KrylovJL_GMRES(), concrete_jac = \
             true)`)."
        )
    )
    LinearAlgebra.mul!(vec(Jv), S, vec(v))
    return nothing
end

"""
    wire_condensed_function(f)

Return the condensed function the global solver is built from: the user's `f` with the Schur
assembler tracked and an analytic `jvp` installed, unless the user supplied a `jvp` already.
Both wrappers share one cell, created here and owned by the cache being built.

The cell is typed on the Jacobian storage rather than left as `Any`, because the Jacobian
cache stores `similar(jac_prototype)`; that keeps the product's `mul!` from dispatching
dynamically on every line-search step. `Nothing` stays in the union to preserve the
"nothing assembled yet" branch.
"""
function wire_condensed_function(f::SciMLBase.NonlinearFunction)
    (SciMLBase.has_jvp(f) || !SciMLBase.has_jac(f)) && return f
    S = f.jac_prototype
    last_S = S === nothing ? Base.RefValue{Any}(nothing) :
        Base.RefValue{Union{Nothing, typeof(S)}}(nothing)
    return SciMLBase.remake(
        f; jac = TrackedSchurAssembly(f.jac, last_S), jvp = SchurJacVecProduct(last_S)
    )
end
