@concrete struct KrylovJᵀJ
    JᵀJ
    Jᵀ
end

__maybe_symmetric(x::KrylovJᵀJ) = x.JᵀJ

isinplace(JᵀJ::KrylovJᵀJ) = isinplace(JᵀJ.Jᵀ)

# Build Jacobian Caches
# function jacobian_caches(alg::AbstractNonlinearSolveAlgorithm, f::F, u, p, ::Val{iip};
#         linsolve_kwargs = (;), lininit::Val{linsolve_init} = Val(true),
#         linsolve_with_JᵀJ::Val{needsJᵀJ} = Val(false)) where {iip, needsJᵀJ, linsolve_init, F}
# du = copy(u)

# if needsJᵀJ
#     JᵀJ, Jᵀfu = __init_JᵀJ(J, _vec(fu), uf, u; f,
#         vjp_autodiff = __get_nonsparse_ad(__getproperty(alg, Val(:vjp_autodiff))),
#         jvp_autodiff = __get_nonsparse_ad(alg.ad))
# else
#     JᵀJ, Jᵀfu = nothing, nothing
# end

# if linsolve_init
#     if alg isa PseudoTransient && J isa SciMLOperators.AbstractSciMLOperator
#         linprob_A = J - inv(convert(eltype(u), alg.alpha_initial)) * I
#     else
#         linprob_A = needsJᵀJ ? __maybe_symmetric(JᵀJ) : J
#     end
#     linsolve = linsolve_caches(linprob_A, needsJᵀJ ? Jᵀfu : fu, du, p, alg;
#         linsolve_kwargs)
# else
#     linsolve = nothing
# end

# return uf, linsolve, J, fu, jac_cache, du, JᵀJ, Jᵀfu
# end

## Special Handling for Scalars
# function jacobian_caches(alg::AbstractNonlinearSolveAlgorithm, f::F, u::Number, p,
#         ::Val{false}; linsolve_with_JᵀJ::Val{needsJᵀJ} = Val(false),
#         kwargs...) where {needsJᵀJ, F}
#     # NOTE: Scalar `u` assumes scalar output from `f`
#     uf = SciMLBase.JacobianWrapper{false}(f, p)
#     return uf, FakeLinearSolveJLCache(u, u), u, zero(u), nothing, u, u, u
# end

__init_JᵀJ(J::Number, args...; kwargs...) = zero(J), zero(J)
function __init_JᵀJ(J::AbstractArray, fu, args...; kwargs...)
    JᵀJ = J' * J
    Jᵀfu = J' * fu
    return JᵀJ, Jᵀfu
end
function __init_JᵀJ(J::StaticArray, fu, args...; kwargs...)
    JᵀJ = MArray{Tuple{size(J, 2), size(J, 2)}, eltype(J)}(undef)
    return JᵀJ, J' * fu
end
function __init_JᵀJ(J::FunctionOperator, fu, uf, u, args...; f = nothing,
        vjp_autodiff = nothing, jvp_autodiff = nothing, kwargs...)
    # FIXME: Proper fix to this requires the FunctionOperator patch
    if f !== nothing && f.vjp !== nothing
        @warn "Currently we don't make use of user provided `jvp`. This is planned to be \
               fixed in the near future."
    end
    autodiff = __concrete_vjp_autodiff(vjp_autodiff, jvp_autodiff, uf)
    Jᵀ = VecJac(uf, u; fu, autodiff)
    JᵀJ_op = SciMLOperators.cache_operator(Jᵀ * J, u)
    JᵀJ = KrylovJᵀJ(JᵀJ_op, Jᵀ)
    Jᵀfu = Jᵀ * fu
    return JᵀJ, Jᵀfu
end

function __concrete_vjp_autodiff(vjp_autodiff, jvp_autodiff, uf)
    if vjp_autodiff === nothing
        if isinplace(uf)
            # VecJac can be only FiniteDiff
            return AutoFiniteDiff()
        else
            # Short circuit if we see that FiniteDiff was used for J computation
            jvp_autodiff isa AutoFiniteDiff && return jvp_autodiff
            # Check if Zygote is loaded then use Zygote else use FiniteDiff
            is_extension_loaded(Val{:Zygote}()) && return AutoZygote()
            return AutoFiniteDiff()
        end
    else
        ad = __get_nonsparse_ad(vjp_autodiff)
        if isinplace(uf) && ad isa AutoZygote
            @warn "Attempting to use Zygote.jl for linesearch on an in-place problem. \
                Falling back to finite differencing."
            return AutoFiniteDiff()
        end
        return ad
    end
end

# jvp fallback scalar
function __gradient_operator(uf, u; autodiff, kwargs...)
    if !(autodiff isa AutoFiniteDiff || autodiff isa AutoZygote)
        _ad = autodiff
        number_ad = ifelse(ForwardDiff.can_dual(eltype(u)), AutoForwardDiff(),
            AutoFiniteDiff())
        if u isa Number
            autodiff = number_ad
        else
            if isinplace(uf)
                autodiff = AutoFiniteDiff()
            else
                autodiff = ifelse(is_extension_loaded(Val{:Zygote}()), AutoZygote(),
                    AutoFiniteDiff())
            end
        end
        if _ad !== nothing && _ad !== autodiff
            @warn "$(_ad) not supported for VecJac. Using $(autodiff) instead."
        end
    end
    return u isa Number ? GradientScalar(uf, u, autodiff) :
           VecJac(uf, u; autodiff, kwargs...)
end

@concrete mutable struct GradientScalar
    uf
    u
    autodiff
end

function Base.:*(jvp::GradientScalar, v::Number)
    if jvp.autodiff isa AutoForwardDiff
        T = typeof(ForwardDiff.Tag(typeof(jvp.uf), typeof(jvp.u)))
        out = jvp.uf(ForwardDiff.Dual{T}(jvp.u, one(v)))
        return ForwardDiff.extract_derivative(T, out)
    elseif jvp.autodiff isa AutoFiniteDiff
        J = FiniteDiff.finite_difference_derivative(jvp.uf, jvp.u, jvp.autodiff.fdtype)
        return J
    else
        error("Only ForwardDiff & FiniteDiff is currently supported.")
    end
end

# Generic Handling of Krylov Methods for Normal Form Linear Solves
function __update_JᵀJ!(cache::AbstractNonlinearSolveCache, J = nothing)
    if !(cache.JᵀJ isa KrylovJᵀJ)
        J_ = ifelse(J === nothing, cache.J, J)
        @bb cache.JᵀJ = transpose(J_) × J_
    end
end

function __update_Jᵀf!(cache::AbstractNonlinearSolveCache, J = nothing)
    if cache.JᵀJ isa KrylovJᵀJ
        @bb cache.Jᵀf = cache.JᵀJ.Jᵀ × cache.fu
    else
        J_ = ifelse(J === nothing, cache.J, J)
        @bb cache.Jᵀf = transpose(J_) × vec(cache.fu)
    end
end
