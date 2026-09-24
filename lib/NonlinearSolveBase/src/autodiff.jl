# Here we determine the preferred AD backend. We have a predefined list of ADs and then
# we select the first one that is available and would work with the problem.

# Ordering is important here. We want to select the first one that is compatible with the
# problem.
const ReverseADs = (
    ADTypes.AutoEnzyme(; mode = EnzymeCore.Reverse),
    ADTypes.AutoZygote(),
    ADTypes.AutoTracker(),
    ADTypes.AutoReverseDiff(),
    ADTypes.AutoFiniteDiff(),
)

const ForwardADs = (
    ADTypes.AutoPolyesterForwardDiff(),
    ADTypes.AutoForwardDiff(),
    ADTypes.AutoEnzyme(; mode = EnzymeCore.Forward),
    ADTypes.AutoFiniteDiff(),
)

"""
    select_forward_mode_autodiff(prob, ad; warn_check_mode = true)

Choose a forward-mode-compatible automatic differentiation backend for `prob`.

If `ad` is an `AbstractADType`, the backend is returned when it is available and compatible
with the problem. If `ad === nothing`, NonlinearSolveBase selects the first available
compatible backend from its preferred forward-mode list.

### Arguments

  - `prob`: A SciML nonlinear problem.
  - `ad`: An ADTypes backend or `nothing`.

### Keyword Arguments

  - `warn_check_mode`: Emit a warning when `ad` is not a forward-mode backend.

### Returns

An ADTypes backend suitable for forward-mode differentiation.

### Examples

```julia
using ADTypes, NonlinearSolveBase, SciMLBase

prob = NonlinearProblem((u, p) -> u^2 - 2, 1.0)
NonlinearSolveBase.select_forward_mode_autodiff(prob, AutoForwardDiff())
```
"""
function select_forward_mode_autodiff(
        prob::AbstractNonlinearProblem, ad::AbstractADType; warn_check_mode::Bool = true
    )
    if warn_check_mode && !(ADTypes.mode(ad) isa ADTypes.ForwardMode) &&
            !(ADTypes.mode(ad) isa ADTypes.ForwardOrReverseMode) &&
            !is_finite_differences_backend(ad)
        @warn lazy"The chosen AD backend $(ad) is not a forward mode AD. Use with caution."

        @warn "The chosen AD backend $(ad) is not a forward mode AD. Use with caution."

    end
    if incompatible_backend_and_problem(prob, ad)
        adₙ = select_forward_mode_autodiff(prob, nothing; warn_check_mode)

        @warn "The chosen AD backend `$(ad)` does not support the chosen problem. This \
               could be because the backend package for the chosen AD isn't loaded. After \
               running autodiff selection detected `$(adₙ)` as a potential forward mode \
               backend."
        return adₙ
    end
    return ad
end

function select_forward_mode_autodiff(
        prob::AbstractNonlinearProblem, ::Nothing;
        warn_check_mode::Bool = true
    )
    idx = findfirst(!Base.Fix1(incompatible_backend_and_problem, prob), ForwardADs)
    idx !== nothing && return ForwardADs[idx]
    throw(ArgumentError("No forward mode AD backend is compatible with the chosen problem. \
                         This could be because no forward mode autodiff backend is loaded \
                         or the loaded backends don't support the problem."))
end

"""
    select_reverse_mode_autodiff(prob, ad; warn_check_mode = true)

Choose a reverse-mode-compatible automatic differentiation backend for `prob`.

If `ad === nothing`, NonlinearSolveBase selects the first available compatible backend from
its preferred reverse-mode list. Finite differencing backends are accepted as explicit
fallback choices.

### Arguments

  - `prob`: A SciML nonlinear problem.
  - `ad`: An ADTypes backend or `nothing`.

### Keyword Arguments

  - `warn_check_mode`: Emit a warning when `ad` is not a reverse-mode backend.

### Returns

An ADTypes backend suitable for reverse-mode differentiation.
"""
function select_reverse_mode_autodiff(
        prob::AbstractNonlinearProblem, ad::AbstractADType; warn_check_mode::Bool = true
    )
    if warn_check_mode && !(ADTypes.mode(ad) isa ADTypes.ReverseMode) &&
            !(ADTypes.mode(ad) isa ADTypes.ForwardOrReverseMode) &&
            !is_finite_differences_backend(ad)
        @warn "The chosen AD backend $(ad) is not a reverse mode AD. Use with caution."
    end
    if incompatible_backend_and_problem(prob, ad)
        adₙ = select_reverse_mode_autodiff(prob, nothing; warn_check_mode)
        @warn "The chosen AD backend `$(ad)` does not support the chosen problem. This \
        could be because the backend package for the chosen AD isn't loaded. After \
        running autodiff selection detected `$(adₙ)` as a potential reverse mode \
        backend."
        return adₙ
    end
    return ad
end

function select_reverse_mode_autodiff(
        prob::AbstractNonlinearProblem, ::Nothing;
        warn_check_mode::Bool = true
    )
    idx = findfirst(!Base.Fix1(incompatible_backend_and_problem, prob), ReverseADs)
    idx !== nothing && return ReverseADs[idx]
    throw(ArgumentError("No reverse mode AD backend is compatible with the chosen problem. \
                         This could be because no reverse mode autodiff backend is loaded \
                         or the loaded backends don't support the problem."))
end

"""
    select_jacobian_autodiff(prob, ad)

Choose an automatic differentiation backend for constructing Jacobians for `prob`.

If `ad === nothing`, NonlinearSolveBase prefers a compatible forward-mode backend that is
not finite differencing, then falls back to compatible reverse-mode or finite-difference
backends.

### Arguments

  - `prob`: A SciML nonlinear problem.
  - `ad`: An ADTypes backend or `nothing`.

### Returns

An ADTypes backend suitable for Jacobian construction.
"""
function select_jacobian_autodiff(prob::AbstractNonlinearProblem, ad::AbstractADType)
    if incompatible_backend_and_problem(prob, ad)
        adₙ = select_jacobian_autodiff(prob, nothing)
        @warn "The chosen AD backend `$(ad)` does not support the chosen problem. This \
        could be because the backend package for the chosen AD isn't loaded. After \
        running autodiff selection detected `$(adₙ)` as a potential forward mode \
        backend."
        return adₙ
    end
    return ad
end

function select_jacobian_autodiff(prob::AbstractNonlinearProblem, ::Nothing)
    idx = findfirst(!Base.Fix1(incompatible_backend_and_problem, prob), ForwardADs)
    idx !== nothing && !is_finite_differences_backend(ForwardADs[idx]) &&
        return ForwardADs[idx]
    idx = findfirst(!Base.Fix1(incompatible_backend_and_problem, prob), ReverseADs)
    idx !== nothing && return ReverseADs[idx]
    throw(ArgumentError("No jacobian AD backend is compatible with the chosen problem. \
                         This could be because no jacobian autodiff backend is loaded \
                         or the loaded backends don't support the problem."))
end

function incompatible_backend_and_problem(
        prob::AbstractNonlinearProblem, ad::AbstractADType
    )
    !DI.check_available(ad) && return true
    SciMLBase.isinplace(prob) && !DI.check_inplace(ad) && return true
    return additional_incompatible_backend_check(prob, ad)
end

additional_incompatible_backend_check(::AbstractNonlinearProblem, ::AbstractADType) = false
function additional_incompatible_backend_check(
        prob::AbstractNonlinearProblem, ::ADTypes.AutoPolyesterForwardDiff
    )
    prob.u0 isa SArray && return true # promotes to a mutable array
    return false
end

is_finite_differences_backend(ad::AbstractADType) = false
is_finite_differences_backend(::ADTypes.AutoFiniteDiff) = true
is_finite_differences_backend(::ADTypes.AutoFiniteDifferences) = true

function nlls_generate_vjp_function(prob::NonlinearLeastSquaresProblem, sol, uu)
    gradient = nlls_generate_gradient_function(prob, sol, uu)
    prob.lb === nothing && prob.ub === nothing && return gradient
    lb = something(prob.lb, -Inf)
    ub = something(prob.ub, Inf)
    # The projected stationarity equation includes the active-bound KKT conditions.
    if SciMLBase.isinplace(prob)
        return @closure (du, u, p) -> begin
            gradient(du, u, p)
            @. du = u - clamp(u - du, lb, ub)
            return nothing
        end
    else
        return @closure (u, p) -> begin
            g = gradient(u, p)
            return @. u - clamp(u - g, lb, ub)
        end
    end
end

function nlls_generate_gradient_function(prob::NonlinearLeastSquaresProblem, sol, uu)
    # First check for custom `vjp` then custom `Jacobian` and if nothing is provided use
    # nested autodiff as the last resort
    return if SciMLBase.has_vjp(prob.f)
        if SciMLBase.isinplace(prob)
            return @closure (
                du, u, p,
            ) -> begin
                resid = Utils.safe_similar(du, length(sol.resid))
                prob.f(resid, u, p)
                prob.f.vjp(du, resid, u, p)
                du .*= 2
                return nothing
            end
        else
            return @closure (
                u, p,
            ) -> begin
                resid = prob.f(u, p)
                g = 2 .* prob.f.vjp(resid, u, p)
                return u isa Number ? g : reshape(g, size(u))
            end
        end
    elseif SciMLBase.has_jac(prob.f)
        if SciMLBase.isinplace(prob)
            return @closure (
                du, u, p,
            ) -> begin
                J = Utils.safe_similar(du, length(sol.resid), length(u))
                prob.f.jac(J, u, p)
                resid = Utils.safe_similar(du, length(sol.resid))
                prob.f(resid, u, p)
                mul!(reshape(du, 1, :), vec(resid)', J, 2, false)
                return nothing
            end
        else
            return @closure (
                u,
                p,
            ) -> begin
                resid = prob.f(u, p)
                J = prob.f.jac(u, p)
                u isa Number && return 2 * LinearAlgebra.dot(J, resid)
                return reshape(2 .* vec(resid)' * J, size(u))
            end
        end
    else
        # For small problems, nesting ForwardDiff is actually quite fast
        autodiff = length(uu) + length(sol.resid) ≥ 50 ?
            select_reverse_mode_autodiff(prob, nothing) : AutoForwardDiff()

        # Use raw (unwrapped) function for nested differentiation to avoid
        # FunctionWrapper type mismatches with nested duals
        raw_f = get_raw_f(prob.f.f)

        # `DI.pullback` with `AutoForwardDiff` is emulated via pushforward into
        # a buffer keyed off `eltype(u)`, which breaks when this vjp closure
        # runs under an outer ForwardDiff layer (the function output becomes
        # Dual while `u` stays Float64). For ForwardDiff we materialize the
        # Jacobian and form `2·J'·r` directly — `DI.jacobian` here dispatches
        # to `ForwardDiff.jacobian`, which handles nested Duals via fresh tags.
        if autodiff isa AutoForwardDiff
            if SciMLBase.isinplace(prob)
                return @closure (du, u, p) -> begin
                    ff = @closure uu -> begin
                        T = promote_type(eltype(uu), eltype(p))
                        r = Utils.safe_similar(uu, T, length(sol.resid))
                        raw_f(r, uu, p)
                        return r
                    end
                    J = DI.jacobian(ff, autodiff, u)
                    mul!(du, J', ff(u), 2, false)
                    return nothing
                end
            else
                return @closure (u, p) -> begin
                    if u isa Number
                        J = DI.derivative(Base.Fix2(raw_f, p), autodiff, u)
                        return 2 * LinearAlgebra.dot(J, raw_f(u, p))
                    end
                    J = DI.jacobian(Base.Fix2(raw_f, p), autodiff, u)
                    return 2 .* (J' * raw_f(u, p))
                end
            end
        end

        if SciMLBase.isinplace(prob)
            return @closure (du, u, p) -> begin
                resid = Utils.safe_similar(du, length(sol.resid))
                raw_f(resid, u, p)
                # Using `Constant` lead to dual ordering issues
                ff = @closure (du, u) -> raw_f(du, u, p)
                resid2 = copy(resid)
                DI.pullback!(ff, resid2, (du,), autodiff, u, (resid,))
                @. du *= 2
                return nothing
            end
        else
            return @closure (u, p) -> begin
                v = raw_f(u, p)
                # Using `Constant` lead to dual ordering issues
                res = only(DI.pullback(Base.Fix2(raw_f, p), autodiff, u, (v,)))
                ArrayInterface.can_setindex(res) || return 2 .* res
                @. res *= 2
                return res
            end
        end
    end
end

implicit_sensitivity_solve(A::Number, B) = A \ B

function implicit_sensitivity_solve(A, B)
    Utils.is_extension_loaded(Val(:LinearSolve)) || return A \ B

    T = promote_type(typeof(oneunit(eltype(A)) / oneunit(eltype(A))), eltype(B))
    u = similar(B, T, size(B))
    lincache = construct_linear_solver(
        nothing, nothing, A, B, u, nothing;
        stats = SciMLBase.NLStats(0, 0, 0, 0, 0), verbose = false
    )
    linres = lincache()
    linres.success || error("Linear solve failed while differentiating a nonlinear solve.")
    return linres.u
end

"""
    nlls_solve_adjoint_dp(prob, sol, p, Δu, save_idxs)

Parameter cotangent `dp = -(∂G/∂p)' · (∂G/∂u)⁻ᵀ · Δu` for the solution `u*` of a
`NonlinearLeastSquaresProblem`, computed by implicit differentiation of the
(projected) stationarity equation `G(u*, p) = 0` built by
[`nlls_generate_vjp_function`](@ref). For problems with `lb`/`ub` bounds this
includes the active-bound KKT conditions: components pinned at a bound have zero
sensitivity while free components satisfy the constrained stationarity system.

`Δu` is the incoming cotangent of `sol.u`. `save_idxs` restricts the sensitivity to
a subset of the state components, matching the `save_idxs` solve keyword.
"""
function nlls_solve_adjoint_dp(prob::NonlinearLeastSquaresProblem, sol, p, Δu, save_idxs)
    # Unwrap AutoSpecializeCallable so the generated stationarity function and the
    # nested differentiation below see the raw callable.
    ad_prob = is_fw_wrapped(prob.f.f) ? @set(prob.f.f = get_raw_f(prob.f.f)) : prob
    G = nlls_generate_vjp_function(ad_prob, sol, sol.u)
    # Prefer ForwardDiff: it composes cleanly with the nested derivative
    # `nlls_generate_gradient_function` may perform internally.
    autodiff = DI.check_available(AutoForwardDiff()) ? AutoForwardDiff() :
        select_jacobian_autodiff(ad_prob, nothing)

    u = sol.u
    G_u = if SciMLBase.isinplace(ad_prob)
        @closure u_ -> begin
            du = Utils.safe_similar(u_, length(u_))
            G(du, u_, p)
            return du
        end
    else
        Base.Fix2(G, p)
    end
    J_u = u isa Number ? DI.derivative(G_u, autodiff, u) :
        DI.jacobian(G_u, autodiff, u)

    G_p = if SciMLBase.isinplace(ad_prob)
        @closure p_ -> begin
            du = Utils.safe_similar(
                u, promote_type(eltype(u), eltype(p_)), length(u)
            )
            G(du, u, p_)
            return du
        end
    else
        Base.Fix1(G, u)
    end
    J_p = if p isa Number
        DI.derivative(G_p, autodiff, p)
    elseif u isa Number
        DI.gradient(G_p, autodiff, p)
    else
        DI.jacobian(G_p, autodiff, p)
    end

    dseed = if u isa Number
        Δu isa AbstractArray ? only(Δu) : Δu
    elseif save_idxs === nothing
        vec(Δu)
    else
        d = zeros(eltype(Δu), length(u))
        d[save_idxs] = Δu isa AbstractArray ? vec(Δu) : Δu
        d
    end
    λ = implicit_sensitivity_solve(J_u', dseed)

    return if p isa Number
        λv = λ isa Number ? λ : vec(λ)
        J_p isa Number ? -(J_p * λv) :
            -LinearAlgebra.dot(vec(J_p), λv isa Number ? [λv] : λv)
    elseif u isa Number
        λs = λ isa Number ? λ : only(λ)
        Utils.safe_reshape(-(λs .* vec(J_p)), size(p))
    else
        Utils.safe_reshape(-(J_p' * vec(λ)), size(p))
    end
end
