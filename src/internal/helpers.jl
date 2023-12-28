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

function evaluate_f!!(prob, fu, u, p)
    if isinplace(prob)
        prob.f(fu, u, p)
        return fu
    else
        return prob.f(u, p)
    end
end

# AutoDiff Selection Functions
struct NonlinearSolveTag end

function ForwardDiff.checktag(::Type{<:ForwardDiff.Tag{<:NonlinearSolveTag, <:T}}, f::F,
        x::AbstractArray{T}) where {T, F}
    return true
end

function get_concrete_forward_ad(autodiff::Union{ADTypes.AbstractForwardMode,
            ADTypes.AbstractFiniteDifferencesMode}, prob, args...; kwargs...)
    return autodiff
end
function get_concrete_forward_ad(autodiff::ADTypes.AbstractADType, prob, args...;
        check_reverse_mode = true, kwargs...)
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
            ADTypes.AbstractFiniteDifferencesMode}, prob, args...; kwargs...)
    return autodiff
end
function get_concrete_reverse_ad(autodiff::Union{AutoZygote, AutoSparseZygote}, prob,
        args...; kwargs...)
    if isinplace(prob)
        @warn "Attempting to use Zygote.jl for inplace problems. Switching to FiniteDiff.\
               Sparsity even if present will be ignored for correctness purposes. Set \
               the reverse ad option to `nothing` to automatically select the best option \
               and exploit sparsity."
        return AutoFiniteDiff() # colorvec confusion will occur if we use FiniteDiff
    end
    return autodiff
end
function get_concrete_reverse_ad(autodiff::ADTypes.AbstractADType, prob, args...;
        check_reverse_mode = true, kwargs...)
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
