# Evaluate the residual function at a given point
function evaluate_f(prob::AbstractNonlinearProblem{uType, iip}, u) where {uType, iip}
    (; f, u0, p) = prob
    if iip
        fu = f.resid_prototype === nothing ? similar(u) :
             promote_type(eltype(u), eltype(f.resid_prototype)).(f.resid_prototype)
        f(fu, u, p)
    else
        fu = f(u, p)
    end
    return fu
end

function evaluate_f!(cache, u, p)
    cache.nf += 1
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

# AutoDiff Selection Functions
struct NonlinearSolveTag end

function ForwardDiff.checktag(::Type{<:ForwardDiff.Tag{<:NonlinearSolveTag, <:T}}, f::F,
        x::AbstractArray{T}) where {T, F}
    return true
end

function get_concrete_forward_ad(autodiff::Union{ADTypes.AbstractForwardMode,
            ADTypes.AbstractFiniteDifferencesMode}, prob, sp::Val{test_sparse} = True,
        args...; kwargs...) where {test_sparse}
    return autodiff
end
function get_concrete_forward_ad(autodiff::ADTypes.AbstractADType, prob,
        sp::Val{test_sparse} = True, args...;
        check_reverse_mode = true, kwargs...) where {test_sparse}
    if check_reverse_mode
        @warn "$(autodiff)::$(typeof(autodiff)) is not a \
               `Abstract(Forward/FiniteDifferences)Mode`. Use with caution." maxlog=1
    end
    return autodiff
end
function get_concrete_forward_ad(autodiff, prob, sp::Val{test_sparse} = True, args...;
        kwargs...) where {test_sparse}
    # TODO: Default to PolyesterForwardDiff for non sparse problems
    if test_sparse
        (; sparsity, jac_prototype) = prob.f
        use_sparse_ad = sparsity !== nothing || jac_prototype !== nothing
    else
        use_sparse_ad = false
    end
    ad = if !ForwardDiff.can_dual(eltype(prob.u0)) # Use Finite Differencing
        use_sparse_ad ? AutoSparseFiniteDiff() : AutoFiniteDiff()
    else
        tag = ForwardDiff.Tag(NonlinearSolveTag(), eltype(prob.u0))
        (use_sparse_ad ? AutoSparseForwardDiff : AutoForwardDiff)(; tag)
    end
    return ad
end

function get_concrete_reverse_ad(autodiff::Union{ADTypes.AbstractReverseMode,
            ADTypes.AbstractFiniteDifferencesMode}, prob, sp::Val{test_sparse} = True,
        args...; kwargs...) where {test_sparse}
    return autodiff
end
function get_concrete_reverse_ad(autodiff::Union{AutoZygote, AutoSparseZygote}, prob,
        sp::Val{test_sparse} = True, args...; kwargs...) where {test_sparse}
    if isinplace(prob)
        @warn "Attempting to use Zygote.jl for inplace problems. Switching to FiniteDiff. \
               Sparsity even if present will be ignored for correctness purposes. Set \
               the reverse ad option to `nothing` to automatically select the best option \
               and exploit sparsity."
        return AutoFiniteDiff() # colorvec confusion will occur if we use FiniteDiff
    end
    return autodiff
end
function get_concrete_reverse_ad(autodiff::ADTypes.AbstractADType, prob,
        sp::Val{test_sparse} = True, args...; check_reverse_mode = true,
        kwargs...) where {test_sparse}
    if check_reverse_mode
        @warn "$(autodiff)::$(typeof(autodiff)) is not a \
               `Abstract(Forward/FiniteDifferences)Mode`. Use with caution." maxlog=1
    end
    return autodiff
end
function get_concrete_reverse_ad(autodiff, prob, sp::Val{test_sparse} = True, args...;
        kwargs...) where {test_sparse}
    # TODO: Default to Enzyme / ReverseDiff for inplace problems?
    if test_sparse
        (; sparsity, jac_prototype) = prob.f
        use_sparse_ad = sparsity !== nothing || jac_prototype !== nothing
    else
        use_sparse_ad = false
    end
    ad = if isinplace(prob) || !is_extension_loaded(Val(:Zygote)) # Use Finite Differencing
        use_sparse_ad ? AutoSparseFiniteDiff() : AutoFiniteDiff()
    else
        use_sparse_ad ? AutoSparseZygote() : AutoZygote()
    end
    return ad
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
    termination_condition !== AbsNormTerminationMode && termination_condition !== nothing &&
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
        du = _vec(similar(_resid))
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
        can_handle_scalar::Val = False, kwargs...)
    Jₚ = JacobianCache(prob, alg, prob.f, fu, u0, prob.p; kwargs...)

    𝓙 = (can_handle_scalar === False && prob.u0 isa Number) ? @closure(u->[Jₚ(u[1])]) : Jₚ

    𝐉 = (can_handle_oop === False && !isinplace(prob)) ?
        @closure((J, u)->copyto!(J, 𝓙(u))) : 𝓙

    return 𝐉
end

# Query Statistics
for stat in (:nsolve, :nfactors, :nsteps, :njacs, :nf)
    fname = Symbol("get_$(stat)")
    @eval @inline $(fname)(cache) = __query_stat(cache, $(Val(stat)))
end

@inline __query_stat(cache, stat::Val) = __direct_query_stat(cache, stat)
@inline @generated function __direct_query_stat(cache::T, ::Val{stat}) where {T, stat}
    hasfield(T, stat) || return :(0)
    return :(__get_data(cache.$(stat)))
end

@inline __get_data(x::Number) = x
@inline __get_data(x::Base.RefValue{Int}) = x[]

function reinit_cache! end
reinit_cache!(cache::Nothing, args...; kwargs...) = nothing
reinit_cache!(cache, args...; kwargs...) = nothing

function __reinit_internal! end
__reinit_internal!(::Nothing, args...; kwargs...) = nothing
__reinit_internal!(cache, args...; kwargs...) = nothing

# Auto-generate some of the helper functions
macro internal_caches(cType, internal_cache_names...)
    return __internal_caches(__source__, __module__, cType, internal_cache_names)
end

function __internal_caches(__source__, __module__, cType, internal_cache_names::Tuple)
    fields = map(name -> :($(__query_stat)(getproperty(cache, $(name)), ST)),
        internal_cache_names)
    callback_caches = map(name -> :($(callback_into_cache!)(cache,
            getproperty(internalcache, $(name)), internalcache, args...)),
        internal_cache_names)
    callbacks_self = map(name -> :($(callback_into_cache!)(internalcache,
            getproperty(internalcache, $(name)))), internal_cache_names)
    reinit_caches = map(name -> :($(reinit_cache!)(getproperty(cache, $(name)),
            args...; kwargs...)), internal_cache_names)
    return esc(quote
        function __query_stat(cache::$(cType), ST::Val{stat}) where {stat}
            val = $(__direct_query_stat)(cache, ST)
            return +($(fields...)) + val
        end
        function __query_stat(cache::$(cType), ST::Val{:nsteps})
            return $(__direct_query_stat)(cache, ST)
        end
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
