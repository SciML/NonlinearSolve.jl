
@concrete mutable struct FakeLinearSolveJLCache
    A
    b
end

@concrete struct FakeLinearSolveJLResult
    cache
    u
end

# Ignores NaN
function __findmin(f, x)
    return findmin(x) do xᵢ
        fx = f(xᵢ)
        return isnan(fx) ? Inf : fx
    end
end

struct NonlinearSolveTag end

function ForwardDiff.checktag(::Type{<:ForwardDiff.Tag{<:NonlinearSolveTag, <:T}}, f::F,
        x::AbstractArray{T}) where {T, F}
    return true
end


@inline value(x) = x
@inline value(x::Dual) = ForwardDiff.value(x)
@inline value(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)



concrete_jac(_) = nothing
concrete_jac(::AbstractNewtonAlgorithm{CJ}) where {CJ} = CJ

_mutable_zero(x) = zero(x)
_mutable_zero(x::SArray) = MArray(x)

_mutable(x) = x
_mutable(x::SArray) = MArray(x)

# __maybe_mutable(x, ::AbstractFiniteDifferencesMode) = _mutable(x)
# The shadow allocated for Enzyme needs to be mutable
__maybe_mutable(x, ::AutoSparseEnzyme) = _mutable(x)
__maybe_mutable(x, _) = x

# Helper function to get value of `f(u, p)`
function evaluate_f(prob::Union{NonlinearProblem{uType, iip},
            NonlinearLeastSquaresProblem{uType, iip}}, u) where {uType, iip}
    @unpack f, u0, p = prob
    if iip
        fu = f.resid_prototype === nothing ? similar(u) : f.resid_prototype
        f(fu, u, p)
    else
        fu = f(u, p)
    end
    return fu
end

function evaluate_f(f::F, u, p, ::Val{iip}; fu = nothing) where {F, iip}
    if iip
        f(fu, u, p)
        return fu
    else
        return f(u, p)
    end
end

function evaluate_f(cache::AbstractNonlinearSolveCache, u, p,
        fu_sym::Val{FUSYM} = Val(nothing)) where {FUSYM}
    cache.stats.nf += 1
    if FUSYM === nothing
        if isinplace(cache)
            cache.prob.f(get_fu(cache), u, p)
        else
            set_fu!(cache, cache.prob.f(u, p))
        end
    else
        if isinplace(cache)
            cache.prob.f(__getproperty(cache, fu_sym), u, p)
        else
            setproperty!(cache, FUSYM, cache.prob.f(u, p))
        end
    end
    return nothing
end

# Concretize Algorithms
function get_concrete_algorithm(alg, prob)
    !hasfield(typeof(alg), :ad) && return alg
    alg.ad isa ADTypes.AbstractADType && return alg

    # Figure out the default AD
    # Now that we have handed trivial cases, we can allow extending this function
    # for specific algorithms
    return __get_concrete_algorithm(alg, prob)
end

function __get_concrete_algorithm(alg, prob)
    @unpack sparsity, jac_prototype = prob.f
    use_sparse_ad = sparsity !== nothing || jac_prototype !== nothing
    ad = if !ForwardDiff.can_dual(eltype(prob.u0))
        # Use Finite Differencing
        use_sparse_ad ? AutoSparseFiniteDiff() : AutoFiniteDiff()
    else
        (use_sparse_ad ? AutoSparseForwardDiff : AutoForwardDiff)(;
            tag = ForwardDiff.Tag(NonlinearSolveTag(), eltype(prob.u0)))
    end
    return set_ad(alg, ad)
end

function init_termination_cache(abstol, reltol, du, u, ::Nothing)
    return init_termination_cache(abstol, reltol, du, u, AbsSafeBestTerminationMode())
end
function init_termination_cache(abstol, reltol, du, u, tc::AbstractNonlinearTerminationMode)
    tc_cache = init(du, u, tc; abstol, reltol)
    return DiffEqBase.get_abstol(tc_cache), DiffEqBase.get_reltol(tc_cache), tc_cache
end

function check_and_update!(cache, fu, u, uprev)
    return check_and_update!(cache.tc_cache, cache, fu, u, uprev)
end
function check_and_update!(tc_cache, cache, fu, u, uprev)
    return check_and_update!(tc_cache, cache, fu, u, uprev,
        DiffEqBase.get_termination_mode(tc_cache))
end
function check_and_update!(tc_cache, cache, fu, u, uprev,
        mode::AbstractNonlinearTerminationMode)
    if tc_cache(fu, u, uprev)
        # Just a sanity measure!
        if isinplace(cache)
            cache.prob.f(get_fu(cache), u, cache.prob.p)
        else
            set_fu!(cache, cache.prob.f(u, cache.prob.p))
        end
        cache.force_stop = true
    end
end
function check_and_update!(tc_cache, cache, fu, u, uprev,
        mode::AbstractSafeNonlinearTerminationMode)
    if tc_cache(fu, u, uprev)
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.Success
            cache.retcode = ReturnCode.Success
        end
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.PatienceTermination
            cache.retcode = ReturnCode.ConvergenceFailure
        end
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.ProtectiveTermination
            cache.retcode = ReturnCode.Unstable
        end
        # Just a sanity measure!
        if isinplace(cache)
            cache.prob.f(get_fu(cache), u, cache.prob.p)
        else
            set_fu!(cache, cache.prob.f(u, cache.prob.p))
        end
        cache.force_stop = true
    end
end
function check_and_update!(tc_cache, cache, fu, u, uprev,
        mode::AbstractSafeBestNonlinearTerminationMode)
    if tc_cache(fu, u, uprev)
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.Success
            cache.retcode = ReturnCode.Success
        end
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.PatienceTermination
            cache.retcode = ReturnCode.ConvergenceFailure
        end
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.ProtectiveTermination
            cache.retcode = ReturnCode.Unstable
        end
        if isinplace(cache)
            copyto!(get_u(cache), tc_cache.u)
            cache.prob.f(get_fu(cache), get_u(cache), cache.prob.p)
        else
            set_u!(cache, tc_cache.u)
            set_fu!(cache, cache.prob.f(get_u(cache), cache.prob.p))
        end
        cache.force_stop = true
    end
end

function __init_low_rank_jacobian(u::StaticArray{S1, T1}, fu::StaticArray{S2, T2},
        ::Val{threshold}) where {S1, S2, T1, T2, threshold}
    T = promote_type(T1, T2)
    fuSize, uSize = Size(fu), Size(u)
    Vᵀ = MArray{Tuple{threshold, prod(uSize)}, T}(undef)
    U = MArray{Tuple{prod(fuSize), threshold}, T}(undef)
    return U, Vᵀ
end
function __init_low_rank_jacobian(u, fu, ::Val{threshold}) where {threshold}
    Vᵀ = similar(u, threshold, length(u))
    U = similar(u, length(fu), threshold)
    return U, Vᵀ
end

@inline __is_ill_conditioned(x::Number) = iszero(x)
@inline __is_ill_conditioned(x::AbstractMatrix) = cond(x) ≥
                                                  inv(eps(real(eltype(x)))^(1 // 2))
@inline __is_ill_conditioned(x::AbstractVector) = any(iszero, x)
@inline __is_ill_conditioned(x) = false

# Non-square matrix
@inline __needs_square_A(_, ::Number) = true
@inline __needs_square_A(alg, _) = LinearSolve.needs_square_A(alg.linsolve)

# Define special concatenation for certain Array combinations
@inline _vcat(x, y) = vcat(x, y)

# LazyArrays for tracing
__zero(x::AbstractArray) = zero(x)
__zero(x) = x
LazyArrays.applied_eltype(::typeof(__zero), x) = eltype(x)
LazyArrays.applied_ndims(::typeof(__zero), x) = ndims(x)
LazyArrays.applied_size(::typeof(__zero), x) = size(x)
LazyArrays.applied_axes(::typeof(__zero), x) = axes(x)


LazyArrays.applied_eltype(::typeof(__safe_inv), x) = eltype(x)
LazyArrays.applied_ndims(::typeof(__safe_inv), x) = ndims(x)
LazyArrays.applied_size(::typeof(__safe_inv), x) = size(x)
LazyArrays.applied_axes(::typeof(__safe_inv), x) = axes(x)

# SparseAD --> NonSparseAD
@inline __get_nonsparse_ad(::AutoSparseForwardDiff) = AutoForwardDiff()
@inline __get_nonsparse_ad(::AutoSparseFiniteDiff) = AutoFiniteDiff()
@inline __get_nonsparse_ad(::AutoSparseZygote) = AutoZygote()
@inline __get_nonsparse_ad(ad) = ad

# Use Symmetric Matrices if known to be efficient
@inline __maybe_symmetric(x) = Symmetric(x)
@inline __maybe_symmetric(x::Number) = x
## LinearSolve with `nothing` doesn't dispatch correctly here
@inline __maybe_symmetric(x::StaticArray) = x
@inline __maybe_symmetric(x::SparseArrays.AbstractSparseMatrix) = x
@inline __maybe_symmetric(x::SciMLOperators.AbstractSciMLOperator) = x

# Diagonal of type `u`
__init_diagonal(u::Number, v) = oftype(u, v)
function __init_diagonal(u::SArray, v)
    u_ = vec(u)
    return Diagonal(ones(typeof(u_)) * v)
end
function __init_diagonal(u, v)
    d = similar(vec(u))
    d .= v
    return Diagonal(d)
end

# Reduce sum
function __sum_JᵀJ!!(y, J)
    if setindex_trait(y) === CanSetindex()
        sum!(abs2, y, J')
        return y
    else
        return sum(abs2, J'; dims = 1)
    end
end

# Alpha for Initial Jacobian Guess
# The values are somewhat different from SciPy, these were tuned to the 23 test problems
@inline function __initial_inv_alpha(α::Number, u, fu, norm::F) where {F}
    return convert(promote_type(eltype(u), eltype(fu)), inv(α))
end
@inline function __initial_inv_alpha(::Nothing, u, fu, norm::F) where {F}
    norm_fu = norm(fu)
    return ifelse(norm_fu ≥ 1e-5, max(norm(u), true) / (2 * norm_fu),
        convert(promote_type(eltype(u), eltype(fu)), true))
end
@inline __initial_inv_alpha(inv_α, α::Number, u, fu, norm::F) where {F} = inv_α
@inline function __initial_inv_alpha(inv_α, α::Nothing, u, fu, norm::F) where {F}
    return __initial_inv_alpha(α, u, fu, norm)
end

@inline function __initial_alpha(α::Number, u, fu, norm::F) where {F}
    return convert(promote_type(eltype(u), eltype(fu)), α)
end
@inline function __initial_alpha(::Nothing, u, fu, norm::F) where {F}
    norm_fu = norm(fu)
    return ifelse(1e-5 ≤ norm_fu ≤ 1e5, max(norm(u), true) / (2 * norm_fu),
        convert(promote_type(eltype(u), eltype(fu)), true))
end
@inline __initial_alpha(α_initial, α::Number, u, fu, norm::F) where {F} = α_initial
@inline function __initial_alpha(α_initial, α::Nothing, u, fu, norm::F) where {F}
    return __initial_alpha(α, u, fu, norm)
end

# Diagonal

@inline __is_complex(::Type{ComplexF64}) = true
@inline __is_complex(::Type{ComplexF32}) = true
@inline __is_complex(::Type{Complex}) = true
@inline __is_complex(::Type{T}) where {T} = false

@inline __reshape(x::Number, args...) = x
@inline __reshape(x::AbstractArray, args...) = reshape(x, args...)
