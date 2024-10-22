# Evaluate the residual function at a given point
function evaluate_f(prob::AbstractNonlinearProblem{uType, iip}, u) where {uType, iip}
    (; f, p) = prob
    if iip
        fu = f.resid_prototype === nothing ? zero(u) : similar(f.resid_prototype)
        f(fu, u, p)
    else
        fu = f(u, p)
    end
    return fu
end

function evaluate_f!(cache, u, p)
    cache.stats.nf += 1
    if isinplace(cache)
        cache.prob.f(get_fu(cache), u, p)
    else
        set_fu!(cache, cache.prob.f(u, p))
    end
end

evaluate_f!!(prob::AbstractNonlinearProblem, fu, u, p) = evaluate_f!!(prob.f, fu, u, p)
function evaluate_f!!(f::NonlinearFunction{iip}, fu, u, p) where {iip}
    if iip
        f(fu, u, p)
        return fu
    end
    return f(u, p)
end

# Callbacks
"""
    callback_into_cache!(cache, internalcache, args...)

Define custom operations on `internalcache` tightly coupled with the calling `cache`.
`args...` contain the sequence of caches calling into `internalcache`.

This unfortunately makes code very tightly coupled and not modular. It is recommended to not
use this functionality unless it can't be avoided (like in [`LevenbergMarquardt`](@ref)).
"""
@inline callback_into_cache!(cache, internalcache, args...) = nothing  # By default do nothing

# Extension Algorithm Helpers
function __test_termination_condition(termination_condition, alg)
    !(termination_condition isa AbsNormTerminationMode) &&
        termination_condition !== nothing &&
        error("`$(alg)` does not support termination conditions!")
end

function __construct_extension_f(prob::AbstractNonlinearProblem; alias_u0::Bool = false,
        can_handle_oop::Val = False, can_handle_scalar::Val = False,
        make_fixed_point::Val = False, force_oop::Val = False)
    if can_handle_oop === False && can_handle_scalar === True
        error("Incorrect Specification: OOP not supported but scalar supported.")
    end

    resid = evaluate_f(prob, prob.u0)
    u0 = can_handle_scalar === True || !(prob.u0 isa Number) ?
         __maybe_unaliased(prob.u0, alias_u0) : [prob.u0]

    fₚ = if make_fixed_point === True
        if isinplace(prob)
            @closure (du, u) -> (prob.f(du, u, prob.p); du .+= u)
        else
            @closure u -> prob.f(u, prob.p) .+ u
        end
    else
        if isinplace(prob)
            @closure (du, u) -> prob.f(du, u, prob.p)
        else
            @closure u -> prob.f(u, prob.p)
        end
    end

    𝐟 = if isinplace(prob)
        u0_size, du_size = size(u0), size(resid)
        @closure (du, u) -> (fₚ(reshape(du, du_size), reshape(u, u0_size)); du)
    else
        if prob.u0 isa Number
            if can_handle_scalar === True
                fₚ
            elseif can_handle_oop === True
                @closure u -> [fₚ(first(u))]
            else
                @closure (du, u) -> (du[1] = fₚ(first(u)); du)
            end
        else
            u0_size = size(u0)
            if can_handle_oop === True
                @closure u -> vec(fₚ(reshape(u, u0_size)))
            else
                @closure (du, u) -> (copyto!(du, fₚ(reshape(u, u0_size))); du)
            end
        end
    end

    𝐅 = if force_oop === True && applicable(𝐟, u0, u0)
        _resid = resid isa Number ? [resid] : _vec(resid)
        du = _vec(zero(_resid))
        @closure u -> begin
            𝐟(du, u)
            return du
        end
    else
        𝐟
    end

    return 𝐅, _vec(u0), (resid isa Number ? [resid] : _vec(resid))
end

function __construct_extension_jac(prob, alg, u0, fu; can_handle_oop::Val = False,
        can_handle_scalar::Val = False, autodiff = nothing, kwargs...)
    autodiff = select_jacobian_autodiff(prob, autodiff)

    Jₚ = JacobianCache(
        prob, alg, prob.f, fu, u0, prob.p; stats = empty_nlstats(), autodiff, kwargs...)

    𝓙 = (can_handle_scalar === False && prob.u0 isa Number) ? @closure(u->[Jₚ(u[1])]) : Jₚ

    𝐉 = (can_handle_oop === False && !isinplace(prob)) ?
        @closure((J, u)->copyto!(J, 𝓙(u))) : 𝓙

    return 𝐉
end

function reinit_cache! end
reinit_cache!(cache::Nothing, args...; kwargs...) = nothing
reinit_cache!(cache, args...; kwargs...) = nothing

function __reinit_internal! end
__reinit_internal!(::Nothing, args...; kwargs...) = nothing
__reinit_internal!(cache, args...; kwargs...) = nothing

# Auto-generate some of the helper functions
macro internal_caches(cType, internal_cache_names...)
    return __internal_caches(cType, internal_cache_names)
end

function __internal_caches(cType, internal_cache_names::Tuple)
    callback_caches = map(
        name -> :($(callback_into_cache!)(
            cache, getproperty(internalcache, $(name)), internalcache, args...)),
        internal_cache_names)
    callbacks_self = map(
        name -> :($(callback_into_cache!)(
            internalcache, getproperty(internalcache, $(name)))),
        internal_cache_names)
    reinit_caches = map(
        name -> :($(reinit_cache!)(getproperty(cache, $(name)), args...; kwargs...)),
        internal_cache_names)
    return esc(quote
        function callback_into_cache!(cache, internalcache::$(cType), args...)
            $(callback_caches...)
        end
        function callback_into_cache!(internalcache::$(cType))
            $(callbacks_self...)
        end
        function reinit_cache!(cache::$(cType), args...; kwargs...)
            $(reinit_caches...)
            $(__reinit_internal!)(cache, args...; kwargs...)
        end
    end)
end
