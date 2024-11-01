# Here we determine the preferred AD backend. We have a predefined list of ADs and then
# we select the first one that is available and would work with the problem.

# Ordering is important here. We want to select the first one that is compatible with the
# problem.
# XXX: Remove this once Enzyme is properly supported on Julia 1.11+
@static if VERSION ≥ v"1.11-"
    const ReverseADs = (
        ADTypes.AutoZygote(),
        ADTypes.AutoTracker(),
        ADTypes.AutoReverseDiff(; compile = true),
        ADTypes.AutoReverseDiff(),
        ADTypes.AutoEnzyme(; mode = EnzymeCore.Reverse),
        ADTypes.AutoFiniteDiff()
    )
else
    const ReverseADs = (
        ADTypes.AutoEnzyme(; mode = EnzymeCore.Reverse),
        ADTypes.AutoZygote(),
        ADTypes.AutoTracker(),
        ADTypes.AutoReverseDiff(; compile = true),
        ADTypes.AutoReverseDiff(),
        ADTypes.AutoFiniteDiff()
    )
end

const ForwardADs = (
    ADTypes.AutoPolyesterForwardDiff(),
    ADTypes.AutoForwardDiff(),
    ADTypes.AutoEnzyme(; mode = EnzymeCore.Forward),
    ADTypes.AutoFiniteDiff()
)

function select_forward_mode_autodiff(
        prob::AbstractNonlinearProblem, ad::AbstractADType; warn_check_mode::Bool = true)
    if warn_check_mode && !(ADTypes.mode(ad) isa ADTypes.ForwardMode) &&
       !(ADTypes.mode(ad) isa ADTypes.ForwardOrReverseMode) &&
       !is_finite_differences_backend(ad)
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

function select_forward_mode_autodiff(prob::AbstractNonlinearProblem, ::Nothing;
        warn_check_mode::Bool = true)
    idx = findfirst(!Base.Fix1(incompatible_backend_and_problem, prob), ForwardADs)
    idx !== nothing && return ForwardADs[idx]
    throw(ArgumentError("No forward mode AD backend is compatible with the chosen problem. \
                         This could be because no forward mode autodiff backend is loaded \
                         or the loaded backends don't support the problem."))
end

function select_reverse_mode_autodiff(
        prob::AbstractNonlinearProblem, ad::AbstractADType; warn_check_mode::Bool = true)
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

function select_reverse_mode_autodiff(prob::AbstractNonlinearProblem, ::Nothing;
        warn_check_mode::Bool = true)
    idx = findfirst(!Base.Fix1(incompatible_backend_and_problem, prob), ReverseADs)
    idx !== nothing && return ReverseADs[idx]
    throw(ArgumentError("No reverse mode AD backend is compatible with the chosen problem. \
                         This could be because no reverse mode autodiff backend is loaded \
                         or the loaded backends don't support the problem."))
end

function select_jacobian_autodiff(prob::AbstractNonlinearProblem, ad::AbstractADType)
    if incompatible_backend_and_problem(prob, ad)
        adₙ = select_jacobian_autodiff(prob, nothing)
        @warn "The chosen AD backend `$(ad)` does not support the chosen problem. This \
               could be because the backend package for the chosen AD isn't loaded. After \
               running autodiff selection detected `$(adₙ)` as a potential jacobian \
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
        prob::AbstractNonlinearProblem, ad::AbstractADType)
    !DI.check_available(ad) && return true
    SciMLBase.isinplace(prob) && !DI.check_inplace(ad) && return true
    return additional_incompatible_backend_check(prob, ad)
end

additional_incompatible_backend_check(::AbstractNonlinearProblem, ::AbstractADType) = false
function additional_incompatible_backend_check(prob::AbstractNonlinearProblem,
        ::ADTypes.AutoReverseDiff{true})
    if SciMLBase.isinplace(prob)
        fu = prob.f.resid_prototype === nothing ? zero(prob.u0) : prob.f.resid_prototype
        return hasbranching(prob.f, fu, prob.u0, prob.p)
    end
    return hasbranching(prob.f, prob.u0, prob.p)
end

is_finite_differences_backend(ad::AbstractADType) = false
is_finite_differences_backend(::ADTypes.AutoFiniteDiff) = true
is_finite_differences_backend(::ADTypes.AutoFiniteDifferences) = true

function nlls_generate_vjp_function(prob::NonlinearLeastSquaresProblem, sol, uu)
    # First check for custom `vjp` then custom `Jacobian` and if nothing is provided use
    # nested autodiff as the last resort
    if SciMLBase.has_vjp(prob.f)
        if SciMLBase.isinplace(prob)
            return @closure (du, u, p) -> begin
                resid = Utils.safe_similar(du, length(sol.resid))
                prob.f.vjp(resid, u, p)
                prob.f.vjp(du, resid, u, p)
                du .*= 2
                return nothing
            end
        else
            return @closure (u, p) -> begin
                resid = prob.f(u, p)
                return reshape(2 .* prob.f.vjp(resid, u, p), size(u))
            end
        end
    elseif SciMLBase.has_jac(prob.f)
        if SciMLBase.isinplace(prob)
            return @closure (du, u, p) -> begin
                J = Utils.safe_similar(du, length(sol.resid), length(u))
                prob.f.jac(J, u, p)
                resid = Utils.safe_similar(du, length(sol.resid))
                prob.f(resid, u, p)
                mul!(reshape(du, 1, :), vec(resid)', J, 2, false)
                return nothing
            end
        else
            return @closure (u, p) -> begin
                return reshape(2 .* vec(prob.f(u, p))' * prob.f.jac(u, p), size(u))
            end
        end
    else
        # For small problems, nesting ForwardDiff is actually quite fast
        autodiff = length(uu) + length(sol.resid) ≥ 50 ?
                   select_reverse_mode_autodiff(prob, nothing) : AutoForwardDiff()

        if SciMLBase.isinplace(prob)
            return @closure (du, u, p) -> begin
                resid = Utils.safe_similar(du, length(sol.resid))
                prob.f(resid, u, p)
                # Using `Constant` lead to dual ordering issues
                ff = @closure (du, u) -> prob.f(du, u, p)
                resid2 = copy(resid)
                DI.pullback!(ff, resid2, (du,), autodiff, u, (resid,))
                @. du *= 2
                return nothing
            end
        else
            return @closure (u, p) -> begin
                v = prob.f(u, p)
                # Using `Constant` lead to dual ordering issues
                res = only(DI.pullback(Base.Fix2(prob.f, p), autodiff, u, (v,)))
                ArrayInterface.can_setindex(res) || return 2 .* res
                @. res *= 2
                return res
            end
        end
    end
end
