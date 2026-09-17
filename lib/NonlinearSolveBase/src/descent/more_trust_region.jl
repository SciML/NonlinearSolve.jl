"""
    MoreTrustRegionDescent(;
        linsolve = nothing, scaling = TrustRegionScaling.None, min_damping_D = 1e-8
    )

Compute the descent direction by solving the trust-region subproblem
`min ‖J δu + fu‖` subject to `‖D δu‖ ≤ Δ` nearly exactly. A safeguarded Newton
iteration finds the damping parameter `λ` such that the least-squares problem

    min ‖ [J; √λ D] δu - [-fu; 0] ‖

— equivalent to the damped normal equations `(JᵀJ + λDᵀD) δu = -Jᵀfu` — has
solution with `‖D δu‖ = Δ`; if the Gauss-Newton step already lies inside the
region it is the subproblem solution and `λ = 0` is returned directly. This is
the algorithm of Moré (MINPACK `lmpar`), also described in Nocedal & Wright
§10.3.

Solving the augmented least-squares system keeps the condition number of `J`
unsquared and handles rank-deficient Jacobians natively: `[J; √λ D]` has full
column rank for every `λ > 0`. When the chosen `linsolve` only supports square
systems the equivalent damped normal equations are used instead, and for
matrix-free (operator) Jacobians the augmented system is applied as a stacked
operator, which Krylov least-squares solvers handle without forming `J`.

Unlike [`Dogleg`](@ref), which follows a two-piece polygonal approximation of
the solution curve, this descent follows the true solution `δu(λ)`, which is
substantially more robust on ill-conditioned least-squares problems. Pair it
with `RadiusUpdateSchemes.More` to get Moré's step-following radius update as
well.

### Keyword Arguments

  - `linsolve`: the linear solver used for the subproblem solves. On the default
    dense path the Jacobian is factorized once per iteration and the damping
    iteration runs against the stored factors (MINPACK `lmpar`/`qrsolv`), so no
    `linsolve` is invoked; on wide Jacobians (`2m ≤ n`) the iteration instead
    works on the `m × m` Gram system `(J D⁻² Jᵀ + λI) w = -fu`. An explicit
    choice is honored through the rectangular augmented system
    `[J; √λD] p = [-fu; 0]`, so it should handle least-squares problems; solvers
    requiring square systems such as `LUFactorization` are routed through the
    normal equations automatically.
  - `scaling`: the diagonal scaling matrix `D`, a [`TrustRegionScaling`](@ref).
    `TrustRegionScaling.None` uses `D = I`; `TrustRegionScaling.Jacobian` uses
    Moré's scaling `Dᵢᵢ = max(Dᵢᵢ, ‖J[:, i]‖)`, which never decreases across
    iterations and makes the trust region scale-covariant.
    `TrustRegionScaling.Auto` engages that same scaling only when the problem
    needs it: if the nonzero column norms of the first Jacobian span a ratio
    `maxⱼ‖J[:, j]‖ / minⱼ‖J[:, j]‖` above `20` the solve proceeds exactly as
    `Jacobian`, and otherwise exactly as `None`. The decision is taken once
    per solve (re-decided after `reinit!`), and a matrix-free Jacobian simply
    never engages. `TrustRegionScaling.Jacobian` requires a concrete Jacobian.
    The legacy symbol spellings `:none`, `:jacobian`, and `:auto` are still
    accepted.
  - `min_damping_D`: lower bound for the entries of `DᵀD` under `Jacobian` or
    active `Auto` scaling.
"""
@concrete struct MoreTrustRegionDescent <: AbstractDescentDirection
    linsolve
    scaling::TrustRegionScaling.T
    # `Val(scaling !== None)`: compile-time flag so `dtd`'s type (and the
    # cache's) stays fixed per algorithm. `Auto` still allocates `dtd` — an
    # inactive gate is `dtd == 1`, the identity — a runtime union would box it
    has_scaling
    min_damping_D
end

function MoreTrustRegionDescent(;
        linsolve = nothing, scaling = TrustRegionScaling.None, min_damping_D = 1.0e-8
    )
    scaling = _more_scaling(scaling)
    return MoreTrustRegionDescent(
        linsolve, scaling, Val(scaling !== TrustRegionScaling.None), min_damping_D
    )
end

supports_trust_region(::MoreTrustRegionDescent) = true

# The dense default path computes a pivoted Householder QR of `J` once per Jacobian
# (MINPACK `qrfac`, stored in `Jbuf`/`Rdiag`/`ipvt`) and then solves each damped
# system by eliminating `√λD` against the triangular `R` with `n` Givens rotations
# (MINPACK `qrsolv`). The sweep produces the upper-triangular `S` with
# `SᵀS = Pᵀ(JᵀJ + λD²)P` directly — the λ-iteration never rebuilds the `(m+n)×n`
# augmented system or a fresh QR, never forms `JᵀJ`, and allocates nothing:
# per-λ work is the O(n²)-order sweep plus triangular solves, and the
# refactorization itself is allocation-free.
#
# With `gram = true` (chosen for `2m ≤ n` Jacobians) the same struct instead holds
# an m-space formulation: `B = J D⁻¹`, `G = B Bᵀ` (`m × m`), and
# `(G + λI) w = -fu`, `D p = Bᵀ w` replace the damped system — equivalent through
# the push-through identity `(BᵀB + λI)⁻¹ Bᵀ = Bᵀ (B Bᵀ + λI)⁻¹`. Each λ trial is
# then an `m × m` Cholesky solve rather than an O(n²) Givens sweep, and the
# refactorization is one `B Bᵀ` (m²n) instead of an mn² QR. The undamped system
# also gains a legitimate solution: when `G` is nonsingular `λ = 0` is the
# minimum-`‖D p‖` Gauss-Newton step, which the padded `n × n` factorization can
# never express (`nsing ≤ m < n` always declines it). Buffers of the inactive
# mode are left empty so `lincache` keeps a single concrete type either way.
mutable struct _MoreLmparWorkspace{T}
    gram::Bool           # selects the m-space Gram formulation over `lmpar`
    Jbuf::Matrix{T}      # `qrfac` output: strict upper is `R`'s off-diagonals;
    # lower trapezoid + diagonal holds the reflector vectors `uⱼ`
    Rdiag::Vector{T}     # `n`: diagonal of `R`; downdated column norms during `qrfac`
    Rw::Matrix{T}        # `n×n`: upper triangle is `R`; strict lower accumulates `Sᵀ`
    ipvt::Vector{Int}    # column permutation: `ipvt[j]` = original index of column `j`
    qtbf::Vector{T}      # `m`: `Qᵀ(-fu)`; refreshed whenever `fu` or `J` changes
    qtbf2::Vector{T}     # same for secondary directions (`idx > 1`, perturbed `fu`)
    sdiag::Vector{T}     # `n`: running bottom-row contents, then diagonal of `S`
    wa::Vector{T}        # `n` rhs/solution scratch; `qrfac` reuses it for norms
    wb::Vector{T}        # `n` scratch for the `q` solve
    wout::Vector{T}      # `n` permuted-solution buffer: `restructure` may alias the
    # returned vector into `cache.p`, so output never shares
    # storage with the scratch vectors
    nsing::Int           # numerical rank of `R` (pivoted QR ⇒ nonincreasing |diag|)
    B::Matrix{T}         # gram: `J D⁻¹`, m×n
    G::Matrix{T}         # gram: `B Bᵀ`, m×m; still read through `Symmetric` —
    # only one triangle is contractually valid from the `A * Aᵀ` product
    C::Matrix{T}         # gram: `G + λI`, consumed in place by `potrf`
    sinv::Vector{T}      # gram: `D⁻¹` diagonal, `n` (all ones when scaling is off)
    w::Vector{T}         # gram: `(G + λI)⁻¹ (-fu)`, m
    z::Vector{T}         # gram: `(G + λI)⁻¹ w` for the derivative term, m
    t::Vector{T}         # gram: `G z` matvec scratch, m
    v::Vector{T}         # gram: `Bᵀ w` — exactly `D p`, n
    stats::NLStats
end

# `normal_form` selects the subproblem formulation: `true` solves the damped
# normal equations `(JᵀJ + λDᵀD) p = -Jᵀfu` (scalars, and `linsolve`s that only
# accept square systems), `false` solves the augmented least-squares system
# `[J; √λD] p = [-fu; 0]` directly (concrete matrices through `linsolve`, and
# matrix-free Jacobians through a stacked operator).
@concrete mutable struct MoreTrustRegionDescentCache <: AbstractDescentCache
    δu
    δus
    lincache
    Jᵀfu        # `Jᵀ fu` for the primary direction
    JᵀJ         # normal-form only
    damped      # normal-form in-place buffer for `JᵀJ + λD²`, else `nothing`
    # These fields stay plain typevars: their allocation gates are compile-time
    # inferable (`normal_form`/`isop`/the `could_lmpar` subset, `alg.has_scaling`),
    # so each cache instantiation pins a concrete type and `=== nothing` checks
    # fold away instead of dispatching on a union
    augmented   # generic dense-path `(m + n) × n` buffer, else `nothing`
    rhs         # generic-path rhs buffer, else `nothing`
    qrhs        # q-solve rhs buffer, else `nothing`
    p           # current trial step
    gn_step     # Gauss-Newton step, reused while `J` and `fu` are unchanged
    gn_norm     # scaled norm `‖D δu_gn‖`; `Inf` when the GN solve failed
    gn_valid::Bool
    dtd         # `Jacobian`/`Auto`-scaling diagonal of `D²`, else `nothing`
    Dp
    D²p
    q
    Jδu         # `J δu` product buffer for the predicted reduction
    λ
    # last completed λ-solve's (λ, ‖D p‖): still a valid data point when the same
    # (J, D, fu) subproblem is retried at a new Δ, so it seeds the next bracket
    λ_bound
    λ_bound_dpnorm
    λ_bound_valid::Bool
    θ
    maxiters::Int
    min_damping_D
    internalnorm
    timer
    preinverted_jacobian <: Union{Val{false}, Val{true}}
    normal_form <: Union{Val{false}, Val{true}}
    jac_convert <: Union{Val{false}, Val{true}}
    op_state
    # `Auto` gate state: undecided until the first Jacobian, then latched;
    # `scaling_active` distinguishes an engaged gate from a latched-off one
    auto_scaling::Bool
    scaling_decided::Bool
    scaling_active::Bool
end

_more_jac_convert(cache::MoreTrustRegionDescentCache) =
    Utils.unwrap_val(cache.jac_convert)

# Operator Jacobians keep a single `FunctionOperator` whose closures read `J`/`λ`
# through this cell: `set_lincache_A!` would `copyto!` a rebuilt operator
# (`FunctionOperator` reports `can_setindex` but defines no `copyto!`), so `A` is
# set once at init — `alias_A = true` keeps it — and the cell carries the updates.
mutable struct _MoreOpState{T}
    J::Any
    λ::T
    Jv::AbstractVector{T}
end

# `[J; √λ I]` as an `(m + n) × n` operator: forward `x ↦ [Jx; √λ x]`, adjoint
# `[y₁; y₂] ↦ Jᵀy₁ + √λ y₂`. Scaling needs column norms, so matrix-free
# Jacobians reject `Jacobian` outright and never engage under `Auto` — the
# damping block is always √λ I here.
function _more_augmented_operator(state::_MoreOpState, u, fu)
    m, n = length(fu), length(u)
    T = promote_type(eltype(u), eltype(fu))
    return FunctionOperator(
        (w, x, _, p, t) -> begin
            mul!(@view(w[1:m]), state.J, x)
            @views @. w[(m + 1):(m + n)] = sqrt(state.λ) * x
            return w
        end,
        Vector{T}(undef, n), Vector{T}(undef, m + n);
        op_adjoint = (w, y, _, p, t) -> begin
            mul!(w, state.J', @view(y[1:m]))
            @views @. w += sqrt(state.λ) * y[(m + 1):(m + n)]
            return w
        end,
        isconstant = true, islinear = true
    )
end

# `x ↦ (JᵀJ + λI)x` for square-only Krylov solvers; `state.Jv` is matvec scratch.
# Symmetric positive definite for λ > 0, so CG/MINRES apply too.
function _more_normal_form_operator(state::_MoreOpState{T}, u) where {T}
    n = length(u)
    nf_op! = (w, x, _, p, t) -> begin
        mul!(state.Jv, state.J, x)
        mul!(w, state.J', state.Jv)
        @. w += state.λ * x
        return w
    end
    return FunctionOperator(
        nf_op!, Vector{T}(undef, n), Vector{T}(undef, n);
        op_adjoint = nf_op!, isconstant = true, islinear = true,
        issymmetric = true, ishermitian = true, isposdef = true
    )
end

# For `n/2 < m < n` the Jacobian is zero-padded to `n` rows: the Householder sweep
# never writes below row `m` (each reflector's tail is zero), so the pad rows stay
# zero through the factorization, `R` comes out rank-deficient (`nsing ≤ m < n`),
# the Gauss–Newton step is declined, and the λ-iteration solves the correct damped
# system `[J; √λD] p = [-fu; 0]` — equivalent to padding the residual with zeros.
# (`2m ≤ n` takes the Gram mode instead.)
function _more_lmpar_workspace(J_::Matrix{T}, stats) where {T}
    m, n = size(J_)
    m2 = max(m, n)
    Jbuf = zeros(T, m2, n)
    copyto!(view(Jbuf, 1:m, :), J_)
    zM, zV = Matrix{T}(undef, 0, 0), Vector{T}(undef, 0)
    ws = _MoreLmparWorkspace{T}(
        false, Jbuf, Vector{T}(undef, n), Matrix{T}(undef, n, n),
        Vector{Int}(undef, n), Vector{T}(undef, m2), Vector{T}(undef, m2),
        Vector{T}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n),
        Vector{T}(undef, n), n,
        zM, zM, zM, zV, zV, zV, zV, zV, stats
    )
    return _more_lmpar_factor!(ws)
end

function _more_lmpar_factor!(ws::_MoreLmparWorkspace{T}, J_) where {T}
    m = size(J_, 1)
    if m == size(ws.Jbuf, 1)
        copyto!(ws.Jbuf, J_)
    else
        Jbuf = ws.Jbuf
        copyto!(view(Jbuf, 1:m, :), J_)
        fill!(view(Jbuf, (m + 1):size(Jbuf, 1), :), zero(T))
    end
    return _more_lmpar_factor!(ws)
end
function _more_lmpar_factor!(ws::_MoreLmparWorkspace{T}) where {T}
    _more_lmpar_qrfac!(ws)
    Rw, Rdiag, A = ws.Rw, ws.Rdiag, ws.Jbuf
    n = size(Rw, 1)
    nsing, tol = n, n * eps(T) * abs(Rdiag[1])
    @inbounds for j in 1:n
        for i in 1:(j - 1)
            Rw[i, j] = A[i, j]
        end
        Rw[j, j] = Rdiag[j]
        nsing == n && abs(Rdiag[j]) <= tol && (nsing = j - 1)
    end
    ws.nsing = nsing
    ws.stats.nfactors += 1
    return ws
end

# In-place Householder QR with column pivoting — a port of MINPACK `qrfac` (real
# case, `pivot = .true.`). `Jbuf` is overwritten: the strict upper trapezoid holds
# `R`'s off-diagonals, and column `j`'s lower trapezoid holds the reflector `uⱼ`
# where `Hⱼ = I - uⱼ uⱼᵀ / uⱼ[j]` and `uⱼ[j] = 1 + xⱼ / ajnorm`. `Rdiag` returns the
# diagonal of `R`, `ipvt` the column permutation; `wa` holds the column norms at
# their last explicit recompute for the downdate drift guard.
function _more_lmpar_qrfac!(ws::_MoreLmparWorkspace{T}) where {T}
    A, rdiag, wa, ipvt = ws.Jbuf, ws.Rdiag, ws.wa, ws.ipvt
    m, n = size(A)
    @inbounds for j in 1:n
        rdiag[j] = wa[j] = BLAS.nrm2(m, pointer(A, (j - 1) * m + 1), 1)
        ipvt[j] = j
    end
    @inbounds for j in 1:n
        kmax = j
        for k in (j + 1):n
            rdiag[k] > rdiag[kmax] && (kmax = k)
        end
        if kmax != j
            for i in 1:m
                A[i, j], A[i, kmax] = A[i, kmax], A[i, j]
            end
            rdiag[kmax] = rdiag[j]
            wa[kmax] = wa[j]
            ipvt[j], ipvt[kmax] = ipvt[kmax], ipvt[j]
        end
        ajnorm = BLAS.nrm2(m - j + 1, pointer(A, (j - 1) * m + j), 1)
        if ajnorm != zero(T)
            A[j, j] < zero(T) && (ajnorm = -ajnorm)
            for i in j:m
                A[i, j] /= ajnorm
            end
            ujj = A[j, j] += one(T)
            for k in (j + 1):n
                s = zero(T)
                for i in j:m
                    s += A[i, j] * A[i, k]
                end
                s /= ujj
                for i in j:m
                    A[i, k] -= s * A[i, j]
                end
                if rdiag[k] != zero(T)
                    t = A[j, k] / rdiag[k]
                    rdiag[k] *= sqrt(max(zero(T), one(T) - t * t))
                    if T(0.05) * abs2(rdiag[k] / wa[k]) <= eps(T)
                        rdiag[k] = wa[k] =
                            BLAS.nrm2(m - j, pointer(A, (k - 1) * m + j + 1), 1)
                    end
                end
            end
        end
        rdiag[j] = -ajnorm
    end
    return nothing
end

# `qtb` = `Qᵀ(-fu)` computed by applying the reflectors packed into `Jbuf` in
# place: `Qᵀ = H_n ⋯ H_1`, so `H₁` acts first. `primary` picks the buffer so a
# perturbed-`fu` solve (`idx > 1`) cannot clobber the primary direction's
# transformed right-hand side.
function _more_lmpar_qtbf!(ws::_MoreLmparWorkspace{T}, fu, primary::Bool) where {T}
    qtbf = primary ? ws.qtbf : ws.qtbf2
    fuv = Utils.safe_vec(fu)
    if length(qtbf) == length(fuv)
        @bb @. qtbf = -fuv
    else
        m = length(fuv)
        @inbounds for i in 1:m
            qtbf[i] = -fuv[i]
        end
        @inbounds for i in (m + 1):length(qtbf)
            qtbf[i] = zero(T)
        end
    end
    A = ws.Jbuf
    m, n = size(A)
    @inbounds for j in 1:n
        ujj = A[j, j]
        ujj == zero(T) && continue
        s = zero(T)
        for i in j:m
            s += A[i, j] * qtbf[i]
        end
        s /= ujj
        for i in j:m
            qtbf[i] -= s * A[i, j]
        end
    end
    return qtbf
end

# Gauss-Newton solve `p = P R⁻¹ qtb`. A rank-deficient `R` leaves the undamped
# system underdetermined — the solution set is a manifold, and MINPACK's
# convention of zeroing the null-space components (the "basic solution") picks
# an arbitrary, generally not minimum-norm, point of it. Report failure instead
# so the caller falls back to the damped iteration, whose `λ > 0` system is
# nonsingular and returns the regularized minimum-norm step.
# `qtb` and `p_out` come off `ws` inside: passing them in separately would make the
# call signature a 3-way union product (`ws`/`qtb`/`wout` each split across the
# `{Float32, Float64}` workspace variants), defeating inference's union-splitting
function _more_lmpar_gn!(ws::_MoreLmparWorkspace{T}, fu, primary::Bool) where {T}
    Rw, wa, nsing, ipvt = ws.Rw, ws.wa, ws.nsing, ws.ipvt
    p_out = ws.wout
    n = size(Rw, 1)
    # `qtbf` must be refreshed even when `R` is rank-deficient: the λ-iteration
    # that runs after this `false` return reads it directly off the workspace
    qtb = _more_lmpar_qtbf!(ws, fu, primary)
    nsing < n && return false
    @inbounds for j in 1:n
        wa[j] = qtb[j]
    end
    @inbounds for j in n:-1:1
        wa[j] /= Rw[j, j]
        tmp = wa[j]
        for i in 1:(j - 1)
            wa[i] -= Rw[i, j] * tmp
        end
    end
    @inbounds for j in 1:n
        p_out[ipvt[j]] = wa[j]
    end
    ws.stats.nsolve += 1
    return true
end

# Solve `[R; √λD̃] z = [qtb; 0]` (with `D̃ = PᵀDP`, `p = Pz`) by eliminating the
# diagonal block into `R` via Givens rotations — a port of MINPACK `qrsolv`. Leaves
# `S` stored as `sdiag` + the strict lower of `Rw` (which holds `S`'s strict upper
# transposed) for the `q` solve; `R`'s upper triangle is preserved in `Rw`.
function _more_lmpar_qrsolv!(
        ws::_MoreLmparWorkspace{T}, qtb, p_out, λ, dtd
    ) where {T}
    Rw, Rdiag, sdiag, wa, ipvt = ws.Rw, ws.Rdiag, ws.sdiag, ws.wa, ws.ipvt
    n = size(Rw, 1)
    sqrtλ = sqrt(λ)
    @inbounds for j in 1:n
        for i in (j + 1):n
            Rw[i, j] = Rw[j, i]
        end
        wa[j] = qtb[j]
    end
    @inbounds for j in 1:n
        dj = dtd === nothing ? sqrtλ : sqrtλ * sqrt(dtd[ipvt[j]])
        if dj != zero(T)
            for k in j:n
                sdiag[k] = zero(T)
            end
            sdiag[j] = dj
            qtbpj = zero(T)
            for k in j:n
                sdiag[k] == zero(T) && continue
                if abs(Rw[k, k]) >= abs(sdiag[k])
                    tanθ = sdiag[k] / Rw[k, k]
                    cosθ = inv(sqrt(one(T) + tanθ * tanθ))
                    sinθ = cosθ * tanθ
                else
                    cotθ = Rw[k, k] / sdiag[k]
                    sinθ = inv(sqrt(one(T) + cotθ * cotθ))
                    cosθ = sinθ * cotθ
                end
                Rw[k, k] = cosθ * Rw[k, k] + sinθ * sdiag[k]
                tmp = cosθ * wa[k] + sinθ * qtbpj
                qtbpj = -sinθ * wa[k] + cosθ * qtbpj
                wa[k] = tmp
                for i in (k + 1):n
                    tmp = cosθ * Rw[i, k] + sinθ * sdiag[i]
                    sdiag[i] = -sinθ * Rw[i, k] + cosθ * sdiag[i]
                    Rw[i, k] = tmp
                end
            end
        end
        sdiag[j] = Rw[j, j]
        Rw[j, j] = Rdiag[j]
    end
    # `sdiag` now holds the diagonal of `S`. A zero entry means `S` is singular
    # (unreachable for `λ > 0` since `√λD` keeps the augmented system full rank —
    # guarded anyway): report the solve as failed rather than zeroing the
    # trailing solution components, which would silently return a wrong `p`.
    @inbounds for j in 1:n
        sdiag[j] == zero(T) && return false
    end
    @inbounds for j in n:-1:1
        s = wa[j]
        for i in (j + 1):n
            s -= Rw[i, j] * wa[i]
        end
        wa[j] = s / sdiag[j]
    end
    @inbounds for j in 1:n
        p_out[ipvt[j]] = wa[j]
    end
    ws.stats.nsolve += 1
    return true
end

# `q = (JᵀJ + λD²)⁻¹ D²p = P S⁻¹ S⁻ᵀ D̃² z` — two triangular solves against the `S`
# left by `_more_lmpar_qrsolv!` (`Sᵀ` forward on its strict lower, `S` back on it)
function _more_lmpar_qsolve!(ws::_MoreLmparWorkspace{T}, dtd, p_vec, q_out) where {T}
    Rw, sdiag, wb, ipvt = ws.Rw, ws.sdiag, ws.wb, ws.ipvt
    n = size(Rw, 1)
    @inbounds for j in 1:n
        l = ipvt[j]
        wb[j] = (dtd === nothing ? p_vec[l] : dtd[l] * p_vec[l])
    end
    @inbounds for j in 1:n
        wb[j] /= sdiag[j]
        tmp = wb[j]
        for i in (j + 1):n
            wb[i] -= Rw[i, j] * tmp
        end
    end
    @inbounds for k in 1:n
        j = n - k + 1
        s = wb[j]
        for i in (j + 1):n
            s -= Rw[i, j] * wb[i]
        end
        wb[j] = s / sdiag[j]
    end
    @inbounds for j in 1:n
        q_out[ipvt[j]] = wb[j]
    end
    ws.stats.nsolve += 1
    return q_out
end

# The safeguarded λ-iteration for the `lmpar` path: `p` aliases `ws.wout` through
# `cache.p` and `q` lands in `ws.wa` — all work is the Givens sweep plus triangular
# solves on the stored `R`/`S`. Scalar setup (`Δ`, the `u₀` bound, initial `λ`) is
# done inside the barrier — `ws`/`dtd` are `Union`-typed fields, so calls through
# them dispatch dynamically and isbits arguments would box on the way in; once
# inside, everything is concrete and the per-λ sweep stays allocation-free.
# `cache.λ` is written in place and only `got_step` is returned.
function _more_lmpar_λloop!(
        cache, ws::_MoreLmparWorkspace{T}, Jᵀfu, dtd, idx1, trust_region
    ) where {T}
    Δ = T(trust_region)
    if dtd === nothing
        u_bound = cache.internalnorm(Jᵀfu) / Δ
    elseif dtd isa Number
        u_bound = abs(Jᵀfu) / (sqrt(dtd) * Δ)
    else
        @bb @. cache.q = Jᵀfu / sqrt(dtd)
        u_bound = cache.internalnorm(cache.q) / Δ
    end
    λ = iszero(cache.λ) ? T(1.0e-3) * u_bound : min(T(cache.λ), u_bound)
    λ = max(λ, eps(T))
    l, uλ = zero(λ), u_bound
    # A rejected step retries the same (J, D, fu) subproblem at a smaller Δ, so the
    # last solve's (λ, ‖Dp‖) is still a valid measurement: ‖Dp(λ)‖ decreasing makes
    # it a one-sided bound on the new root and `λ ‖Dp‖ / Δ` a secant seed near it
    if idx1 && cache.λ_bound_valid
        if cache.λ_bound_dpnorm >= Δ
            l = max(l, cache.λ_bound)
            λ = min(cache.λ_bound * (cache.λ_bound_dpnorm / Δ), u_bound)
        else
            uλ = min(uλ, cache.λ_bound)
            λ = min(λ, uλ)
        end
    end
    uλ = max(uλ, λ)
    cache.λ = λ
    qtb = idx1 ? ws.qtbf : ws.qtbf2
    got_step = false
    pos_bound = l > zero(l)
    ϕ_prev, λ_prev = zero(λ), λ
    for i in 1:cache.maxiters
        if !_more_lmpar_qrsolv!(ws, qtb, ws.wout, λ, dtd) || !_all_finite(ws.wout)
            l = max(l, λ)
            λ *= 10
            uλ = max(uλ, λ)
            continue
        end
        cache.p = Utils.restructure(cache.p, ws.wout)
        p = cache.p
        got_step = true
        cache.λ = λ

        Dp = _more_Dp!(cache, dtd, p)
        dpnorm = cache.internalnorm(Dp)
        ϕ = dpnorm - Δ
        if idx1
            cache.λ_bound = λ
            cache.λ_bound_dpnorm = dpnorm
            cache.λ_bound_valid = true
        end
        (abs(ϕ) <= cache.θ * Δ || i == cache.maxiters) && break
        # MINPACK's `parl == 0` exit: with no positive-ϕ evaluation yet, a λ decrease
        # that grew ‖Dp‖ by less than the convergence tolerance means the boundary
        # is unreachable — the rank-deficient hard case — so keep the step in hand
        (
            !pos_bound && ϕ_prev < zero(ϕ) && ϕ <= ϕ_prev + cache.θ * Δ &&
                λ <= λ_prev
        ) && break
        ϕ > zero(ϕ) && (pos_bound = true)
        ϕ_prev, λ_prev = ϕ, λ

        D²p = _more_D2p!(cache, dtd, p)
        _more_lmpar_qsolve!(ws, dtd, p, ws.wa)
        if !_all_finite(ws.wa)
            l = max(l, λ)
            λ *= 10
            uλ = max(uλ, λ)
            continue
        end
        λ, l, uλ = _more_update_λ(
            λ, ϕ, Δ, Utils.safe_dot(Dp, Dp), Utils.safe_dot(D²p, ws.wa), l, uλ
        )
    end
    return got_step
end

function _more_gram_workspace(J_::Matrix{T}, dtd, stats) where {T}
    m, n = size(J_)
    zM, zV = Matrix{T}(undef, 0, 0), Vector{T}(undef, 0)
    ws = _MoreLmparWorkspace{T}(
        true, zM, zV, zM, Vector{Int}(undef, 0), zV, zV, zV, zV, zV,
        Vector{T}(undef, n), n,
        Matrix{T}(undef, m, n), Matrix{T}(undef, m, m), Matrix{T}(undef, m, m),
        Vector{T}(undef, n), Vector{T}(undef, m), Vector{T}(undef, m),
        Vector{T}(undef, m), Vector{T}(undef, n), stats
    )
    return _more_gram_factor!(ws, J_, dtd)
end

function _more_gram_factor!(ws::_MoreLmparWorkspace{T}, J_, dtd) where {T}
    B, sinv = ws.B, ws.sinv
    m, n = size(J_)
    if dtd === nothing
        copyto!(B, J_)
        fill!(sinv, one(T))
    else
        @inbounds for j in 1:n
            s = inv(sqrt(dtd[j]))
            sinv[j] = s
            for i in 1:m
                B[i, j] = J_[i, j] * s
            end
        end
    end
    mul!(ws.G, B, transpose(B))
    ws.stats.nfactors += 1
    return ws
end

# `w ← -fu`, the right-hand side of the damped `m`-system
function _more_gram_rhs!(w, fu)
    fuv = Utils.safe_vec(fu)
    @bb @. w = -fuv
    return w
end

# `p = D⁻¹ v` with `v = Bᵀ w` already computed (`sinv` is all-ones unscaled);
# `wout` plays the output-buffer role
function _more_gram_step!(ws::_MoreLmparWorkspace)
    @inbounds for j in eachindex(ws.wout)
        ws.wout[j] = ws.v[j] * ws.sinv[j]
    end
    return ws.wout
end

# Undamped `G w = -fu` — the minimum-`‖D p‖` Gauss-Newton step. A numerically
# singular `G` declines the step (the analogue of `nsing < n` in `_more_lmpar_gn!`)
# so the λ-iteration regularizes it instead.
function _more_gram_gn!(ws::_MoreLmparWorkspace, fu)
    copyto!(ws.C, Symmetric(ws.G))
    F = LinearAlgebra.cholesky!(Symmetric(ws.C); check = false)
    LinearAlgebra.issuccess(F) || return false
    _more_gram_rhs!(ws.w, fu)
    ldiv!(F, ws.w)
    mul!(ws.v, transpose(ws.B), ws.w)
    _all_finite(ws.v) || return false
    _more_gram_step!(ws)
    ws.stats.nsolve += 1
    return true
end

# The `lmpar`-loop analogue for the Gram formulation — same φ definition,
# bracketing, θ-tolerance, and ×10 retry — but each λ trial is one `m × m`
# Cholesky of `G + λI`. The Newton term is computable in m-space: with
# `z = (G + λI)⁻¹ w`, `D q = (BᵀB + λI)⁻¹ D p = (BᵀB + λI)⁻¹ Bᵀ w = Bᵀ z`, hence
# `pᵀD²q = (Bᵀw)ᵀ(Bᵀz) = wᵀ G z` — evaluated as `wᵀ(Gz)` rather than the
# equivalent `wᵀw - λ wᵀz`, which cancels badly when `w` is near `null(G)`.
function _more_gram_λloop!(
        cache, ws::_MoreLmparWorkspace{T}, Jᵀfu, fu, dtd, trust_region
    ) where {T}
    Δ = T(trust_region)
    if dtd === nothing
        u_bound = cache.internalnorm(Jᵀfu) / Δ
    else
        @bb @. cache.q = Jᵀfu / sqrt(dtd)
        u_bound = cache.internalnorm(cache.q) / Δ
    end
    λ = iszero(cache.λ) ? T(1.0e-3) * u_bound : min(T(cache.λ), u_bound)
    GS = Symmetric(ws.G)
    # `G + λI` must stay numerically definite for `potrf`; floor mirrors the
    # normal-form path's `eps * maxdiag(JᵀJ) / min(dtd)` regularization limit
    λ = max(λ, eps(T) * _more_maxdiag(GS))
    l, uλ = zero(λ), max(u_bound, λ)
    cache.λ = λ
    B, C, w, z, t, v = ws.B, ws.C, ws.w, ws.z, ws.t, ws.v
    m = size(C, 1)
    got_step = false
    for i in 1:cache.maxiters
        copyto!(C, GS)
        @inbounds for k in 1:m
            C[k, k] += λ
        end
        F = LinearAlgebra.cholesky!(Symmetric(C); check = false)
        ok = LinearAlgebra.issuccess(F)
        if ok
            _more_gram_rhs!(w, fu)
            ldiv!(F, w)
            mul!(v, transpose(B), w)
            ok = _all_finite(v)
        end
        if !ok
            # λ is numerically too small to regularize the system; the analytic
            # bound u₀ assumes exact arithmetic, so λ is allowed to outgrow it
            l = max(l, λ)
            λ *= 10
            uλ = max(uλ, λ)
            continue
        end
        ws.stats.nsolve += 1
        _more_gram_step!(ws)
        cache.p = Utils.restructure(cache.p, ws.wout)
        p = cache.p
        got_step = true
        cache.λ = λ

        Dp = v
        ϕ = cache.internalnorm(Dp) - Δ
        (abs(ϕ) <= cache.θ * Δ || i == cache.maxiters) && break

        copyto!(z, w)
        ldiv!(F, z)
        if !_all_finite(z)
            l = max(l, λ)
            λ *= 10
            uλ = max(uλ, λ)
            continue
        end
        ws.stats.nsolve += 1
        mul!(t, GS, z)
        λ, l, uλ = _more_update_λ(
            λ, ϕ, Δ, Utils.safe_dot(Dp, Dp), Utils.safe_dot(w, t), l, uλ
        )
    end
    return got_step
end

# Generic-path twin of `_more_lmpar_λloop!` — the LinearSolve-based damped/q solves
# for the normal-form, operator, and explicit-`linsolve` cases. Keeping the loop
# (and the scalar setup) behind a barrier narrows `dtd` to its runtime type, which
# keeps `λ_of_p` (and so the `extras` NamedTuple fields) concretely inferred.
function _more_generic_λloop!(
        cache, lincache, J_, Jᵀfu, fu, u, dtd, idx1, trust_region, kwargs
    )
    T = promote_type(eltype(u), eltype(fu))
    Δ = T(trust_region)
    if dtd === nothing
        u_bound = cache.internalnorm(Jᵀfu) / Δ
    elseif dtd isa Number
        u_bound = abs(Jᵀfu) / (sqrt(dtd) * Δ)
    else
        @bb @. cache.q = Jᵀfu / sqrt(dtd)
        u_bound = cache.internalnorm(cache.q) / Δ
    end
    λ = iszero(cache.λ) ? T(1.0e-3) * u_bound : min(T(cache.λ), u_bound)
    l, uλ = zero(λ), u_bound
    # Same warm start as `_more_lmpar_λloop!`: the last solve's (λ, ‖Dp‖) still
    # measures the current (J, D, fu) subproblem when only Δ changed
    if idx1 && cache.λ_bound_valid
        if cache.λ_bound_dpnorm >= Δ
            l = max(l, cache.λ_bound)
            λ = min(cache.λ_bound * (cache.λ_bound_dpnorm / Δ), u_bound)
        else
            uλ = min(uλ, cache.λ_bound)
            λ = min(λ, uλ)
        end
    end
    # Below ~eps·maxdiag(JᵀJ)/min(diag DᵀD) the normal-equations factorization cannot
    # succeed; the augmented system stays full rank but still clamps λ off 0 to keep
    # the `Dp/√λ` right-hand side of the q-solve finite
    λ = if normal_form(cache)
        max(λ, eps(T) * _more_maxdiag(cache.JᵀJ) / _more_mindtd(dtd))
    else
        max(λ, eps(T))
    end
    uλ = max(uλ, λ)
    cache.λ = λ
    got_step = false
    pos_bound = l > zero(l)
    ϕ_prev, λ_prev = zero(λ), λ
    for i in 1:cache.maxiters
        linres = _more_damped_solve(cache, lincache, J_, Jᵀfu, fu, λ, u, kwargs)
        if !linres.success || !_all_finite(linres.u)
            # λ is numerically too small to regularize the system; the analytic
            # bound u₀ assumes exact arithmetic, so λ is allowed to outgrow it
            l = max(l, λ)
            λ *= 10
            uλ = max(uλ, λ)
            continue
        end
        p = linres.u
        if normal_form(cache)
            if p isa Number
                cache.p = -p
            else
                @bb @. cache.p = -p
            end
        else
            cache.p = Utils.restructure(cache.p, p)
        end
        p = cache.p
        got_step = true
        cache.λ = λ

        Dp = _more_Dp!(cache, dtd, p)
        dpnorm = cache.internalnorm(Dp)
        ϕ = dpnorm - Δ
        if idx1
            cache.λ_bound = λ
            cache.λ_bound_dpnorm = dpnorm
            cache.λ_bound_valid = true
        end
        (abs(ϕ) <= cache.θ * Δ || i == cache.maxiters) && break
        # Same `parl == 0` stagnation exit as `_more_lmpar_λloop!`
        (
            !pos_bound && ϕ_prev < zero(ϕ) && ϕ <= ϕ_prev + cache.θ * Δ &&
                λ <= λ_prev
        ) && break
        ϕ > zero(ϕ) && (pos_bound = true)
        ϕ_prev, λ_prev = ϕ, λ

        D²p = _more_D2p!(cache, dtd, p)
        qres = _more_q_solve(cache, lincache, D²p, Dp, λ, u, kwargs)
        if !qres.success || !_all_finite(qres.u)
            l = max(l, λ)
            λ *= 10
            uλ = max(uλ, λ)
            continue
        end
        λ, l, uλ = _more_update_λ(
            λ, ϕ, Δ, Utils.safe_dot(Dp, Dp), Utils.safe_dot(D²p, qres.u), l, uλ
        )
    end
    return got_step
end

InternalAPI.reinit!(::_MoreLmparWorkspace, args...; kwargs...) = nothing

# `linsolve_kwargs` keys live in the `NamedTuple`'s type parameters, so the
# whitelist folds to a literal at compile time — this keeps `could_lmpar` (and
# hence `lincache`'s typevar) compile-time decidable, which is what makes
# `@inferred init` hold. A plain `issubset(keys(kw), ...)` does not fold on
# non-empty tuples and would leave `could_lmpar` a runtime `Bool`.
@generated function _more_lmpar_kwargs_ok(::NamedTuple{K}) where {K}
    return issubset(K, (:verbose, :abstol, :reltol))
end

function InternalAPI.init(
        prob::AbstractNonlinearProblem, alg::MoreTrustRegionDescent, J, fu, u; stats,
        pre_inverted::Val = Val(false), linsolve_kwargs = (;),
        abstol = nothing, reltol = nothing, internalnorm::F = L2_NORM,
        shared::Val = Val(1), timer = get_timer_output(), kwargs...
    ) where {F}
    length(fu) != length(u) &&
        @assert !Utils.unwrap_val(pre_inverted) "Precomputed Inverse for Non-Square Jacobian doesn't make sense."
    has_scaling = Utils.unwrap_val(alg.has_scaling)
    auto_scaling = alg.scaling === TrustRegionScaling.Auto
    isop = J === nothing || J isa AbstractSciMLOperator
    if isop
        Utils.unwrap_val(pre_inverted) &&
            throw(ArgumentError("`MoreTrustRegionDescent` cannot invert a matrix-free \
                                 Jacobian; use `concrete_jac = true` or a different \
                                 descent algorithm."))
        alg.scaling === TrustRegionScaling.Jacobian &&
            throw(ArgumentError("`scaling = TrustRegionScaling.Jacobian` needs \
                                 column norms of a concrete Jacobian; use \
                                 `TrustRegionScaling.None` or \
                                 `TrustRegionScaling.Auto` for matrix-free \
                                 problems."))
        if needs_concrete_A(alg.linsolve)
            # a convertible operator (`MatrixOperator`-style `jac_prototype`) is
            # materialized once, like factorization solvers do lazily; a genuinely
            # matrix-free Jacobian cannot feed a concrete-only solver
            (J isa AbstractSciMLOperator && isconvertible(J)) ||
                throw(ArgumentError("`MoreTrustRegionDescent` on a matrix-free Jacobian \
                                     needs a `linsolve` that acts on operators (the \
                                     default Krylov selection is one); \
                                     `$(alg.linsolve)` requires a concrete system."))
        end
    end
    jac_convert = isop && needs_concrete_A(alg.linsolve)

    J_ = Utils.unwrap_val(pre_inverted) ? inv(J) : J
    jac_convert && (J_ = convert(AbstractMatrix, J_); isop = false)
    T = promote_type(eltype(u), eltype(fu))

    # Statics go through the normal equations as well: `SMatrix \ vector` on a
    # rectangular augmented system hits a missing `adjoint` in StaticArrays' QR,
    # while the small square `JᵀJ` is cheap to form. Operator Jacobians take the
    # normal-form path when `linsolve` needs a square system: `JᵀJ + λI` is then
    # applied as an operator.
    # Operator Jacobians with the default `linsolve` take the normal-form operator:
    # `x ↦ (JᵀJ + λI) x` is square and symmetric positive definite, so the default
    # Krylov selection works on it — the non-square `[J; √λI]` operator hits broken
    # `DefaultLinearSolver`/least-squares-Krylov dispatches on older LinearSolve
    # Concrete sparse(-structured) Jacobians with the default `linsolve` take the
    # normal form too: the augmented `[J; √λD]` system would force an (m + n) × n
    # sparse factorization per λ, while the n × n normal equations get a
    # symbolic-reusing sparse LU through `default_spd_linsolve`'s `nothing`
    # fallback. `has_sparsestruct` is the ArrayInterface trait for CSC/banded/
    # (block-)diagonal layouts — decidable on `J_`'s type, no SparseArrays import.
    normal_form = u isa Number || J isa Number || J_ isa StaticArray ||
        needs_square_A(alg.linsolve, u) || (isop && alg.linsolve === nothing) ||
        (alg.linsolve === nothing && ArrayInterface.has_sparsestruct(J_))
    # Dense Jacobians with the default `linsolve` take the MINPACK `lmpar`/`qrsolv`
    # path: one pivoted QR of `J` per Jacobian, then per-λ Givens elimination of
    # `√λD` against the triangular factor — no per-λ refactorization, no `JᵀJ`, no
    # allocations. An explicit `linsolve`, or `linsolve_kwargs` beyond the always-
    # forwarded `:verbose`/`:abstol`/`:reltol`, keeps the generic LinearSolve-driven
    # path so the user's solver choice is honored. The gate is compile-time
    # inferable so `lincache`'s typevar binds a single concrete type per
    # specialization (`@inferred init` holds and no union dispatch survives into
    # `solve!`); `m < n` Jacobians take the same `_MoreLmparWorkspace` in Gram
    # mode when `2m ≤ n`, and are otherwise handled inside it by zero-padding.
    could_lmpar = !normal_form && !isop && alg.linsolve === nothing &&
        _more_lmpar_kwargs_ok(linsolve_kwargs) &&
        (T === Float32 || T === Float64) && J_ isa Matrix{T}

    @bb δu = zero(u)
    δus = Utils.unwrap_val(shared) ≤ 1 ? nothing : map(2:Utils.unwrap_val(shared)) do i
            @bb δu_ = zero(u)
    end

    if u isa Number
        p, gn_step, Dp, D²p, q = zero(u), zero(u), zero(u), zero(u), zero(u)
        Jᵀfu, Jδu, augmented, rhs, qrhs = zero(u), zero(fu), nothing, nothing, nothing
    else
        # `p`, `Jᵀfu`, ... live in the vectorized linear-algebra space: for a
        # matrix-shaped `u`/`fu` they are `length(u)`/`length(fu)` vectors while `δu`
        # keeps the state shape (`restructure` maps between them at the boundary)
        @bb p = similar(Utils.safe_vec(u))
        @bb gn_step = similar(Utils.safe_vec(u))
        @bb Dp = similar(Utils.safe_vec(u))
        @bb D²p = similar(Utils.safe_vec(u))
        @bb q = similar(Utils.safe_vec(u))
        @bb Jᵀfu = similar(Utils.safe_vec(u))
        Jδu = if fu isa Number
            zero(fu)
        else
            @bb Jδu_ = similar(Utils.safe_vec(fu))
            Jδu_
        end
        if normal_form
            augmented = rhs = qrhs = nothing
        else
            m, n = length(fu), length(u)
            # `could_lmpar` leaves `augmented` as `nothing`: the `lmpar` path
            # never assembles the stacked system. Every mutable concrete `J_`
            # otherwise gets an in-place buffer on `J_`'s device; the diagonal
            # block is written by broadcast so GPU arrays (no fast scalar
            # indexing) take the same path
            augmented = if !could_lmpar && J_ isa AbstractMatrix &&
                    ArrayInterface.can_setindex(J_)
                similar(J_, T, m + n, n)
            else
                nothing
            end
            # `similar(safe_vec(u), …)` keeps the rhs on the state's device
            rhs = similar(Utils.safe_vec(u), T, m + n)
            qrhs = similar(Utils.safe_vec(u), T, m + n)
        end
    end

    op_state = isop ?
        _MoreOpState(J_, zero(T), similar(Utils.safe_vec(u), T, length(u))) : nothing
    if normal_form
        JᵀJ = if isop
            nothing
        elseif J_ isa Number
            abs2(J_)
        else
            transpose(J_) * J_
        end
        dtd = if has_scaling
            auto_scaling ? _more_auto_dtd(u, JᵀJ, alg.min_damping_D) :
                _more_scaling_init(JᵀJ, u, alg.min_damping_D)
        else
            nothing
        end
        damped = _more_damped_buffer(JᵀJ, T)
        A0 = if op_state !== nothing
            _more_normal_form_operator(op_state, u)
        else
            A0 = _more_damped_system(JᵀJ, T(1.0e-3), dtd, damped, u)
            A0 isa AbstractMatrix ? Utils.maybe_symmetric(A0) : A0
        end
        Jᵀfu0 = J_ isa Number ? J_ * fu : transpose(J_) * Utils.safe_vec(fu)
        linsolve = alg.linsolve === nothing ? default_spd_linsolve(A0) : alg.linsolve
        lincache = construct_linear_solver(
            alg, linsolve, A0, Utils.safe_vec(Jᵀfu0), Utils.safe_vec(u), prob.p;
            stats, abstol, reltol, linsolve_kwargs...
        )
    elseif could_lmpar
        JᵀJ = damped = nothing
        dtd = if has_scaling
            auto_scaling ? _more_auto_dtd(u, J_, alg.min_damping_D) :
                _more_scaling_init(J_, u, alg.min_damping_D, Val(:columns))
        else
            nothing
        end
        # `2m ≤ n` switches the subproblem to the m-space Gram mode: the m×m
        # Cholesky per λ is cheaper than the n×n Givens sweep and, unlike the
        # padded factorization, can express the undamped minimum-norm step. For
        # `n/2 < m < n` the padded `lmpar` mode is retained — its O(n²) sweep
        # beats an O(m³) Cholesky when `m` approaches `n`
        lincache = if 2 * size(J_, 1) <= size(J_, 2)
            _more_gram_workspace(J_, dtd, stats)
        else
            _more_lmpar_workspace(J_, stats)
        end
    else
        JᵀJ = damped = nothing
        dtd = if has_scaling
            auto_scaling ? _more_auto_dtd(u, J_, alg.min_damping_D) :
                _more_scaling_init(J_, u, alg.min_damping_D, Val(:columns))
        else
            nothing
        end
        A0 = if op_state !== nothing
            _more_augmented_operator(op_state, u, fu)
        else
            _more_augmented_system(J_, T(1.0e-3), dtd, augmented, fu, u)
        end
        b0 = _more_augmented_rhs!(rhs, fu)
        lincache = construct_linear_solver(
            alg, alg.linsolve, A0, b0, Utils.safe_vec(u), prob.p;
            stats, abstol, reltol, linsolve_kwargs...
        )
    end

    return MoreTrustRegionDescentCache(
        δu, δus, lincache, Jᵀfu, JᵀJ, damped, augmented, rhs, qrhs,
        p, gn_step, T(Inf), false, dtd, Dp, D²p, q, Jδu, zero(T),
        zero(T), zero(T), false, T(1.0e-4), 10, T(alg.min_damping_D),
        internalnorm, timer, pre_inverted,
        Val(normal_form), Val(jac_convert), op_state,
        auto_scaling, !auto_scaling, has_scaling && !auto_scaling
    )
end

function InternalAPI.reinit!(cache::MoreTrustRegionDescentCache, args...; kwargs...)
    InternalAPI.reinit!(cache.lincache, args...; kwargs...)
    cache.λ = zero(cache.λ)
    cache.gn_valid = false
    cache.λ_bound_valid = false
    if cache.auto_scaling
        # `Auto` re-decides on the next solve's first Jacobian
        cache.scaling_decided = false
        cache.scaling_active = false
        cache.dtd isa AbstractVector && fill!(cache.dtd, one(eltype(cache.dtd)))
        cache.dtd isa Number && (cache.dtd = one(cache.dtd))
    else
        cache.dtd isa AbstractVector && fill!(cache.dtd, cache.min_damping_D)
        cache.dtd isa Number && (cache.dtd = cache.min_damping_D)
    end
    return
end

function NonlinearSolveBase.callback_into_cache!(
        topcache, cache::MoreTrustRegionDescentCache, args...
    )
    # An accepted step changed `fu`, so the cached Gauss-Newton step, `Jᵀfu`, and the
    # λ bound are stale even when the Jacobian was reused
    tr_cache = Utils.safe_getproperty(topcache, Val(:trustregion_cache))
    if tr_cache isa AbstractTrustRegionMethodCache &&
            NonlinearSolveBase.last_step_accepted(tr_cache)
        cache.gn_valid = cache.λ_bound_valid = false
    end
    return NonlinearSolveBase.callback_into_cache!(cache, cache.lincache)
end

# `dtd` stores the squared diagonal of `D`; `D δu` and `D² δu` go through these helpers
function _more_Dp!(cache, dtd::AbstractVector, p)
    @bb @. cache.Dp = sqrt(dtd) * p
    return cache.Dp
end
_more_Dp!(cache, ::Nothing, p) = p
_more_Dp!(cache, dtd::Number, p::Number) = sqrt(dtd) * p
function _more_D2p!(cache, dtd::AbstractVector, p)
    @bb @. cache.D²p = dtd * p
    return cache.D²p
end
_more_D2p!(cache, ::Nothing, p) = p
_more_D2p!(cache, dtd::Number, p::Number) = dtd * p

_more_scaled_norm(cache, dtd, p) = cache.internalnorm(_more_Dp!(cache, dtd, p))
function _more_scaled_norm(cache, dtd, p::Number)
    dtd === nothing && return abs(p)
    return sqrt(dtd) * abs(p)
end

# `Dᵢᵢ = max(Dᵢᵢ, ‖J[:, i]‖)`: read off the `JᵀJ` diagonal under normal form, the
# column norms of `J` directly otherwise — the same numbers either way
function _more_scaling_init(JᵀJ::AbstractMatrix, u, min_damping)
    dtd = _more_dtd_buffer(u)
    if ArrayInterface.fast_scalar_indexing(JᵀJ)
        @inbounds for i in axes(JᵀJ, 1)
            dtd[i] = max(abs(JᵀJ[i, i]), min_damping)
        end
    else
        dtd .= max.(abs.(diag(JᵀJ)), min_damping)
    end
    return dtd
end
function _more_scaling_init(J::AbstractMatrix, u, min_damping, ::Val{:columns})
    dtd = _more_dtd_buffer(u)
    if ArrayInterface.fast_scalar_indexing(J)
        @inbounds for j in axes(J, 2)
            dtd[j] = max(abs2(norm(@view(J[:, j]))), min_damping)
        end
    else
        dtd .= max.(vec(sum(abs2, J; dims = 1)), min_damping)
    end
    return dtd
end
_more_scaling_init(JᵀJ::Number, u, min_damping) = max(JᵀJ, min_damping)

# `similar(u)` keeps SVector-shaped states on a static diagonal; everything mutable
# goes through the usual concrete vector
function _more_dtd_buffer(u)
    dtd = similar(vec(u))
    ArrayInterface.can_setindex(dtd) ||
        (dtd = Vector{eltype(dtd)}(undef, length(dtd)))
    return dtd
end

function _more_scaling_update!(cache, JᵀJ::AbstractMatrix)
    dtd = cache.dtd
    dtd === nothing && return
    if ArrayInterface.fast_scalar_indexing(JᵀJ)
        @inbounds for i in axes(JᵀJ, 1)
            dtd[i] = max(dtd[i], abs(JᵀJ[i, i]), cache.min_damping_D)
        end
    else
        dtd .= max.(dtd, abs.(diag(JᵀJ)), cache.min_damping_D)
    end
    return
end
function _more_scaling_update!(cache, J::AbstractMatrix, ::Val{:columns})
    dtd = cache.dtd
    dtd === nothing && return
    if ArrayInterface.fast_scalar_indexing(J)
        @inbounds for j in axes(J, 2)
            dtd[j] = max(
                dtd[j], abs2(norm(@view(J[:, j]))), cache.min_damping_D
            )
        end
    else
        dtd .= max.(dtd, vec(sum(abs2, J; dims = 1)), cache.min_damping_D)
    end
    return
end
# operators reach here under `Auto` only with the gate already latched off, so
# `dtd` is either `nothing` or identity
_more_scaling_update!(cache, ::Any, ::Val{:columns}) = nothing
_more_scaling_update!(cache, ::AbstractSciMLOperator) = nothing
function _more_scaling_update!(cache, JᵀJ::Number)
    dtd = cache.dtd
    dtd === nothing && return
    cache.dtd = max(dtd, abs(JᵀJ), cache.min_damping_D)
    return
end

# Workspace for `JᵀJ + λD²` reused across the λ-iteration. Anything that supports
# fast scalar `setindex!` gets a plain `similar` buffer; everything else — sparse
# matrices included — leaves `damped === nothing` and `_more_damped_system` forms
# `JᵀJ + λD²` out of place per λ, keeping the same sparsity pattern each time.
function _more_damped_buffer(JᵀJ, ::Type{T}) where {T}
    if JᵀJ isa AbstractMatrix && ArrayInterface.can_setindex(JᵀJ) &&
            ArrayInterface.fast_scalar_indexing(JᵀJ)
        return similar(JᵀJ)
    end
    return nothing
end

# `Auto` engages Moré scaling when the first Jacobian's nonzero column norms
# span a ratio `maxⱼ‖J[:, j]‖ / minⱼ‖J[:, j]‖ > _MORE_AUTO_SCALING_THRESHOLD`;
# the decision then latches for the rest of the solve
const _MORE_AUTO_SCALING_THRESHOLD = 20

# `one(_more_scaling_init(...))` keeps the scalar `dtd` on the same type the
# `Jacobian` branch would return, so `init` stays concrete
_more_auto_dtd(u, JᵀJ::Number, min_damping) =
    one(_more_scaling_init(JᵀJ, u, min_damping))
function _more_auto_dtd(u, J, min_damping)
    dtd = _more_dtd_buffer(u)
    fill!(dtd, one(eltype(dtd)))
    return dtd
end

function _more_update_scaling!(cache, J, kind...)
    if cache.auto_scaling && !cache.scaling_decided
        return _more_auto_decide!(cache, J, kind...)
    end
    cache.scaling_active && _more_scaling_update!(cache, J, kind...)
    return nothing
end

function _more_auto_decide!(cache, JᵀJ::AbstractMatrix)
    dtd = cache.dtd
    if ArrayInterface.fast_scalar_indexing(JᵀJ)
        @inbounds for i in axes(JᵀJ, 1)
            dtd[i] = abs(JᵀJ[i, i])
        end
    else
        dtd .= abs.(diag(JᵀJ))
    end
    return _more_auto_latch!(cache, dtd)
end
function _more_auto_decide!(cache, J::AbstractMatrix, ::Val{:columns})
    dtd = cache.dtd
    if ArrayInterface.fast_scalar_indexing(J)
        @inbounds for j in axes(J, 2)
            dtd[j] = abs2(norm(@view(J[:, j])))
        end
    else
        dtd .= vec(sum(abs2, J; dims = 1))
    end
    return _more_auto_latch!(cache, dtd)
end
# operators expose no column norms, and scalar/`Val(:columns)`-incompatible
# Jacobians have no ratio to gate on — `Auto` stays off
_more_auto_decide!(cache, ::Any, ::Val{:columns}) = _more_auto_off!(cache)
_more_auto_decide!(cache, ::Union{Number, AbstractSciMLOperator}) =
    _more_auto_off!(cache)

function _more_auto_off!(cache)
    cache.scaling_active = false
    return cache.scaling_decided = true
end

function _more_auto_latch!(cache, dtd::AbstractVector)
    if _more_needs_scaling(dtd)
        cache.scaling_active = true
        if ArrayInterface.fast_scalar_indexing(dtd)
            @inbounds for i in eachindex(dtd)
                dtd[i] = max(dtd[i], cache.min_damping_D)
            end
        else
            dtd .= max.(dtd, cache.min_damping_D)
        end
    else
        fill!(dtd, one(eltype(dtd)))
    end
    cache.scaling_decided = true
    return nothing
end

# `dtd` holds the squared column norms `sⱼ = ‖J[:, j]‖²`: the gate's ratio on the
# norms is `hi / lo > τ²` here, and zero columns are excluded from both ends so
# they neither divide-by-zero nor force the gate
function _more_needs_scaling(dtd::AbstractVector{T}) where {T}
    nz, lo, hi = 0, typemax(T), zero(T)
    if ArrayInterface.fast_scalar_indexing(dtd)
        @inbounds for i in eachindex(dtd)
            si = dtd[i]
            si == zero(T) && continue
            nz += 1
            si < lo && (lo = si)
            si > hi && (hi = si)
        end
    else
        nz = count(>(zero(T)), dtd)
        lo = minimum(s -> s > zero(T) ? s : typemax(T), dtd)
        hi = maximum(dtd)
    end
    return nz ≥ 2 && hi / lo > T(_MORE_AUTO_SCALING_THRESHOLD)^2
end

function _more_damped_system(JᵀJ::Number, λ, dtd, damped, u)
    return JᵀJ + λ * (dtd === nothing ? one(JᵀJ) : dtd)
end
function _more_damped_system(JᵀJ::AbstractMatrix, λ, dtd, damped, u)
    if damped === nothing
        return JᵀJ + λ * (dtd === nothing ? LinearAlgebra.I : Diagonal(dtd))
    end
    copyto!(damped, JᵀJ)
    @simd ivdep for i in axes(damped, 1)
        @inbounds damped[i, i] += λ * (dtd === nothing ? one(λ) : dtd[i])
    end
    return damped
end
# `[J; √λ D]` assembled into `buf` for mutable dense `J` (rewritten in full each
# λ since the factorization may consume it); `vcat` keeps sparse `J` sparse
function _more_augmented_system(J_::AbstractMatrix, λ, dtd, buf, fu, u)
    if buf === nothing
        return vcat(J_, _more_damping_block(J_, λ, dtd))
    end
    m, n = size(J_)
    copyto!(@view(buf[1:m, :]), J_)
    blk = @view(buf[(m + 1):(m + n), :])
    fill!(blk, zero(eltype(buf)))
    if ArrayInterface.fast_scalar_indexing(buf)
        @inbounds for j in 1:n
            blk[j, j] = sqrt(λ) * (dtd === nothing ? one(λ) : sqrt(dtd[j]))
        end
    else
        blk .+= _more_damping_block(J_, λ, dtd)
    end
    return buf
end

# `√λ D` as an `n × n` bottom block matching `J`'s storage flavor
# `vcat` promotes `Diagonal` to sparse under a sparse `J`, so only the static
# case needs an explicit `SMatrix` block to keep the system fully static
function _more_damping_block(J_::StaticArray, λ, dtd)
    n = size(J_, 2)
    T = promote_type(eltype(J_), typeof(λ))
    return dtd === nothing ? sqrt(λ) * SMatrix{n, n, T}(LinearAlgebra.I) :
        SMatrix{n, n, T}(Diagonal(SVector{n, T}(sqrt.(λ .* dtd))))
end
function _more_damping_block(J_::AbstractMatrix, λ, dtd)
    n = size(J_, 2)
    T = promote_type(eltype(J_), typeof(λ))
    # `similar(J_)` keeps the diagonal's storage on `J_`'s device
    d = dtd === nothing ? fill!(similar(J_, T, n), one(T)) : sqrt.(dtd)
    return sqrt(λ) * Diagonal(d)
end

function _more_augmented_rhs!(rhs, fu)
    m = length(fu)
    @views rhs[1:m] .= .-Utils.safe_vec(fu)
    fill!(@view(rhs[(m + 1):end]), zero(eltype(rhs)))
    return rhs
end

function _more_augmented_qrhs!(qrhs, Dp, λ, m)
    fill!(@view(qrhs[1:m]), zero(eltype(qrhs)))
    @views @. qrhs[(m + 1):end] = Dp / sqrt(λ)
    return qrhs
end

_all_finite(x::Number) = isfinite(x)
_all_finite(x) = all(isfinite, x)

# `eps * maxdiag(JᵀJ) / min(diag D²)`: below this the damped system cannot factorize as
# positive definite, so the safeguarded iteration must not start or retry below it
function _more_maxdiag(JᵀJ::AbstractMatrix)
    if ArrayInterface.fast_scalar_indexing(JᵀJ)
        m = zero(eltype(JᵀJ))
        @inbounds for i in axes(JᵀJ, 1)
            m = max(m, abs(JᵀJ[i, i]))
        end
        return m
    else
        return maximum(abs, diag(JᵀJ))
    end
end
_more_maxdiag(JᵀJ::Number) = abs(JᵀJ)
# `JᵀJ` is `nothing` for operator Jacobians: the λ floor is only needed against
# factorization failure, which Krylov solves do not hit
_more_maxdiag(::Union{Nothing, AbstractSciMLOperator}) = true
_more_mindtd(::Nothing) = 1
_more_mindtd(dtd::AbstractVector) = minimum(dtd)
_more_mindtd(dtd::Number) = dtd

# Moré's safeguarded Newton update on φ(λ) = ‖D δu‖ - Δ: tighten the bracket [l, u] on
# λ* and fall back to a guarded step if the Newton iterate leaves it
function _more_update_λ(λ, ϕ, Δ, pᵀD²p, pᵀD²q, l, u)
    ϕ < 0 ? (u = λ) : (l = λ)
    λ += ϕ / Δ * pᵀD²p / pᵀD²q
    l <= λ <= u || (λ = max(l + 0.01 * (u - l), sqrt(l * u)))
    return λ, l, u
end

function InternalAPI.solve!(
        cache::MoreTrustRegionDescentCache, J, fu, u, idx::Val = Val(1);
        skip_solve::Bool = false, new_jacobian::Bool = true,
        trust_region = nothing, kwargs...
    )
    T = promote_type(eltype(u), eltype(fu))
    δu = SciMLBase.get_du(cache, idx)
    # every return path must build the same `extras` NamedTuple shape, or the
    # unioned `DescentResult` return type forces a boxed `getproperty` on the
    # caller's unconditional `descent_result.extras` read. NaN sentinels rather
    # than zeros: `λ = 0` reads as "undamped Gauss-Newton" downstream
    # (`RadiusUpdateSchemes.More` expands the radius on `iszero(λ)`), which a
    # failed solve must not claim.
    empty_extras = (;
        λ = T(NaN), δuJᵀJδu = T(NaN), predicted_reduction = T(NaN),
        step_norm = T(NaN),
    )
    # A Moré direction is Δ-dependent, so there is no honest direction to return
    # without solving — report failure instead of the stale buffer plus
    # fabricated extras
    skip_solve && return DescentResult(δu, missing, false, true, empty_extras)
    @assert trust_region !== nothing "`trust_region` must be specified for \
        `MoreTrustRegionDescent`."
    Δ = T(trust_region)
    idx1 = idx === Val(1)
    dtd = cache.dtd
    J_ = preinverted_jacobian(cache) ? inv(J) : J
    _more_jac_convert(cache) && (J_ = convert(AbstractMatrix, J_))
    cache.op_state !== nothing && (cache.op_state.J = J_)

    if new_jacobian && idx1
        if normal_form(cache) && cache.op_state === nothing
            if J_ isa Number
                cache.JᵀJ = abs2(J_)
            else
                @bb cache.JᵀJ = transpose(J_) × J_
            end
            _more_update_scaling!(cache, cache.JᵀJ)
        else
            # `dtd` is refreshed before factoring because the Gram workspace folds
            # `D⁻¹` into `B`/`G` at factor time; the `lmpar` path reads `dtd`
            # inside its λ-solves, so the order is immaterial for it
            _more_update_scaling!(cache, J_, Val(:columns))
            # narrow the `lincache` union once so the factorization call splits
            # onto the concrete workspace types
            if (lincache = cache.lincache) isa _MoreLmparWorkspace
                lincache.gram ? _more_gram_factor!(lincache, J_, dtd) :
                    _more_lmpar_factor!(lincache, J_)
            end
        end
        cache.gn_valid = false
        cache.λ_bound_valid = false
    elseif idx1 && cache.auto_scaling && !cache.scaling_decided
        # a reused first Jacobian still settles a pending `Auto` gate
        if normal_form(cache) && cache.op_state === nothing
            _more_auto_decide!(cache, cache.JᵀJ)
        else
            _more_auto_decide!(cache, J_, Val(:columns))
        end
    end

    # Gauss-Newton step: when it lies inside the region it is the subproblem solution.
    # On a rejected retry (`new_jacobian = false`) both `J` and `fu` are unchanged, so
    # the cached step is reused without an extra factorization. Secondary directions
    # (`idx > 1`, e.g. under `GeodesicAcceleration`) evaluate at a perturbed `fu`, so
    # they recompute `Jᵀfu` and the GN step into scratch without touching the cache.
    Jᵀfu, gn_step, gn_norm = if idx1 && cache.gn_valid
        (cache.Jᵀfu, cache.gn_step, cache.gn_norm)
    else
        Jᵀfu = if idx1
            if J_ isa Number
                cache.Jᵀfu = J_ * fu
            else
                @bb cache.Jᵀfu = transpose(J_) × vec(fu)
            end
        else
            J_ isa Number ? J_ * fu : transpose(J_) * Utils.safe_vec(fu)
        end
        # Jᵀfu = 0 is a stationary point of the model: p = 0 solves the subproblem for
        # every Δ, and the λ bracket below degenerates to the empty interval (0, 0].
        if iszero(cache.internalnorm(Jᵀfu))
            δu = Utils.restructure(δu, zero(cache.p))
            set_du!(cache, δu, idx)
            extras = _more_extras(cache, J_, δu, zero(T), dtd)
            return DescentResult(δu, missing, true, true, extras)
        end
        gn_buf = idx1 ? cache.gn_step : cache.p
        linres_u, linres_ok = if (lincache = cache.lincache) isa _MoreLmparWorkspace
            if lincache.gram
                (lincache.wout, _more_gram_gn!(lincache, fu))
            else
                (lincache.wout, _more_lmpar_gn!(lincache, fu, idx1))
            end
        else
            linres = _more_gn_solve(
                cache, lincache, J_, Jᵀfu, fu, gn_buf, u, kwargs
            )
            (linres.u, linres.success)
        end
        if linres_ok && _all_finite(linres_u)
            if gn_buf isa AbstractArray && ArrayInterface.can_setindex(gn_buf)
                if normal_form(cache)
                    @bb @. gn_buf = -linres_u
                else
                    # `linres.u` is `Any` through the unioned `lincache` call —
                    # `copyto!` on the concrete buffer avoids broadcasting `Any`
                    copyto!(gn_buf, linres_u)
                end
                gn = gn_buf
            else
                gn = normal_form(cache) ?
                    (linres_u isa Number ? -linres_u : .-linres_u) : linres_u
                gn = Utils.restructure(gn_buf, gn)
            end
            idx1 && (cache.gn_step = gn)
            gn_norm = _more_scaled_norm(cache, dtd, gn)
        else
            gn = nothing
            gn_norm = T(Inf)
        end
        idx1 && (cache.gn_valid = true; cache.gn_norm = gn_norm)
        (Jᵀfu, gn, gn_norm)
    end

    if gn_norm <= Δ
        δu = Utils.restructure(δu, gn_step)
        set_du!(cache, δu, idx)
        extras = _more_extras(cache, J_, δu, zero(T), dtd)
        return DescentResult(δu, missing, true, true, extras)
    end

    # Moré's safeguarded Newton iteration on the damping parameter (MINPACK `lmpar`):
    # λ* ∈ (0, u₀] with u₀ = ‖D⁻¹ Jᵀfu‖ / Δ, since ‖D δu(λ)‖ ≤ ‖D⁻¹ Jᵀfu‖ / λ.
    # Both loops leave λ* in `cache.λ` and return whether a step was produced.
    @static_timeit cache.timer "more iteration" begin
        got_step = if (lincache = cache.lincache) isa _MoreLmparWorkspace
            if lincache.gram
                _more_gram_λloop!(cache, lincache, Jᵀfu, fu, dtd, trust_region)
            else
                _more_lmpar_λloop!(cache, lincache, Jᵀfu, dtd, idx1, trust_region)
            end
        else
            _more_generic_λloop!(
                cache, lincache, J_, Jᵀfu, fu, u, dtd, idx1, trust_region, kwargs
            )
        end
    end
    λ_of_p = cache.λ

    if !got_step
        set_du!(cache, δu, idx)
        return DescentResult(δu, missing, false, false, empty_extras)
    end

    δu = Utils.restructure(δu, cache.p)
    set_du!(cache, δu, idx)
    extras = _more_extras(cache, J_, δu, λ_of_p, dtd)
    return DescentResult(δu, missing, true, true, extras)
end

# Undamped `min ‖Jp + fu‖` — the `λ = 0` augmented system. Normal form instead
# solves `JᵀJ p = Jᵀfu` on the same square cache (the step is negated after).
function _more_gn_solve(cache, lincache, J_, Jᵀfu, fu, gn_buf, u, kwargs)
    if cache.op_state !== nothing
        cache.op_state.λ = zero(cache.op_state.λ)
        b = normal_form(cache) ? Utils.safe_vec(Jᵀfu) :
            _more_augmented_rhs!(cache.rhs, fu)
        return lincache(; b, linu = Utils.safe_vec(gn_buf), kwargs...)
    end
    if normal_form(cache)
        A0 = cache.JᵀJ isa AbstractMatrix ? Utils.maybe_symmetric(cache.JᵀJ) :
            cache.JᵀJ
        return lincache(;
            A = A0, b = Utils.safe_vec(Jᵀfu), linu = Utils.safe_vec(gn_buf),
            reuse_A_if_factorization = false, kwargs...
        )
    end
    A = _more_augmented_system(
        J_, zero(promote_type(eltype(u), eltype(fu))), cache.dtd, cache.augmented, fu, u
    )
    b = _more_augmented_rhs!(cache.rhs, fu)
    return lincache(;
        A, b, linu = Utils.safe_vec(gn_buf),
        reuse_A_if_factorization = false, kwargs...
    )
end

# Damped solve: normal form `(JᵀJ + λD²) p = Jᵀfu` (negated after); augmented
# `min ‖[J; √λD] p - [-fu; 0]‖`. The `lmpar` path runs its own Givens sweep
# inside `solve!` against the stored `R`.
function _more_damped_solve(cache, lincache, J_, Jᵀfu, fu, λ, u, kwargs)
    if cache.op_state !== nothing
        cache.op_state.λ = λ
        b = normal_form(cache) ? Utils.safe_vec(Jᵀfu) :
            _more_augmented_rhs!(cache.rhs, fu)
        return lincache(; b, linu = Utils.safe_vec(cache.p), kwargs...)
    end
    if normal_form(cache)
        A = _more_damped_system(cache.JᵀJ, λ, cache.dtd, cache.damped, u)
        A = A isa AbstractMatrix ? Utils.maybe_symmetric(A) : A
        return lincache(;
            A, b = Utils.safe_vec(Jᵀfu), linu = Utils.safe_vec(cache.p),
            reuse_A_if_factorization = false, kwargs...
        )
    end
    A = _more_augmented_system(J_, λ, cache.dtd, cache.augmented, fu, u)
    b = _more_augmented_rhs!(cache.rhs, fu)
    return lincache(;
        A, b, linu = Utils.safe_vec(cache.p),
        reuse_A_if_factorization = false, kwargs...
    )
end

# `q = (JᵀJ + λD²)⁻¹ D²p`, so `pᵀD²q` feeds the Newton update. Normal form solves
# the square system directly; the augmented path uses `A \ [0; Dp/√λ]`, since
# `Aᵀ [0; Dp/√λ] = D²p` for `A = [J; √λD]`.
function _more_q_solve(cache, lincache, D²p, Dp, λ, u, kwargs)
    if cache.op_state !== nothing || normal_form(cache)
        b = normal_form(cache) ? Utils.safe_vec(D²p) :
            _more_augmented_qrhs!(cache.qrhs, Dp, λ, length(cache.rhs) - length(cache.p))
        return lincache(;
            b, linu = Utils.safe_vec(cache.q),
            reuse_A_if_factorization = true, kwargs...
        )
    end
    b = _more_augmented_qrhs!(cache.qrhs, Dp, λ, length(cache.rhs) - length(cache.p))
    return lincache(;
        b, linu = Utils.safe_vec(cache.q),
        reuse_A_if_factorization = true, kwargs...
    )
end

# `extras` carries `λ` and the scaled step norm for `RadiusUpdateSchemes.More`, and the
# MINPACK-form predicted reduction `½‖Jδu‖² + λ‖Dδu‖²`, which avoids the cancellation
# of `-gᵀδu - ½δuᵀJᵀJδu` at a subproblem solution for ill-conditioned `J`
function _more_extras(cache, J_, δu, λ, dtd)
    if J_ isa Number
        δuJᵀJδu = abs2(J_ * δu)
    elseif cache.Jδu isa AbstractVector
        @bb cache.Jδu = J_ × vec(δu)
        δuJᵀJδu = Utils.safe_dot(cache.Jδu, cache.Jδu)
    else
        δuJᵀJδu = Utils.safe_dot(J_ * Utils.safe_vec(δu), J_ * Utils.safe_vec(δu))
    end
    Dp = _more_Dp!(cache, dtd, δu)
    predicted_reduction = δuJᵀJδu / 2 + λ * Utils.safe_dot(Dp, Dp)
    return (; λ, δuJᵀJδu, predicted_reduction, step_norm = cache.internalnorm(Dp))
end
