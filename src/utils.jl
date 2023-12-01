const DEFAULT_NORM = DiffEqBase.NONLINEARSOLVE_DEFAULT_NORM

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

"""
    default_adargs_to_adtype(; chunk_size = Val{0}(), autodiff = Val{true}(),
        standardtag = Val{true}(), diff_type = Val{:forward})

Construct the AD type from the arguments. This is mostly needed for compatibility with older
code.

!!! warning

    `chunk_size`, `standardtag`, `diff_type`, and `autodiff::Union{Val, Bool}` are
    deprecated and will be removed in v3. Update your code to directly specify
    `autodiff=<ADTypes>`.
"""
function default_adargs_to_adtype(; chunk_size = missing, autodiff = nothing,
        standardtag = missing, diff_type = missing)
    # If using the new API short circuit
    autodiff === nothing && return nothing
    autodiff isa ADTypes.AbstractADType && return autodiff

    # Deprecate the old version
    if chunk_size !== missing || standardtag !== missing || diff_type !== missing ||
       autodiff !== missing
        Base.depwarn("`chunk_size`, `standardtag`, `diff_type`, \
            `autodiff::Union{Val, Bool}` kwargs have been deprecated and will be removed \
             in v3. Update your code to directly specify autodiff=<ADTypes>",
            :default_adargs_to_adtype)
    end
    chunk_size === missing && (chunk_size = Val{0}())
    standardtag === missing && (standardtag = Val{true}())
    diff_type === missing && (diff_type = Val{:forward}())
    autodiff === missing && (autodiff = Val{true}())

    ad = _unwrap_val(autodiff)
    # We don't really know the typeof the input yet, so we can't use the correct tag!
    ad &&
        return AutoForwardDiff{_unwrap_val(chunk_size), NonlinearSolveTag}(NonlinearSolveTag())
    return AutoFiniteDiff(; fdtype = diff_type)
end

"""
    value_derivative(f, x)

Compute `f(x), d/dx f(x)` in the most efficient way.
"""
function value_derivative(f::F, x::R) where {F, R}
    T = typeof(ForwardDiff.Tag(f, R))
    out = f(ForwardDiff.Dual{T}(x, one(x)))
    ForwardDiff.value(out), ForwardDiff.extract_derivative(T, out)
end

@inline value(x) = x
@inline value(x::Dual) = ForwardDiff.value(x)
@inline value(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)

@inline _vec(v) = vec(v)
@inline _vec(v::Number) = v
@inline _vec(v::AbstractVector) = v

@inline _restructure(y, x) = restructure(y, x)
@inline _restructure(y::Number, x::Number) = x

DEFAULT_PRECS(W, du, u, p, t, newW, Plprev, Prprev, cachedata) = nothing, nothing

function dolinsolve(precs::P, linsolve::FakeLinearSolveJLCache; A = nothing,
    linu = nothing, b = nothing, du = nothing, p = nothing, weight = nothing,
    cachedata = nothing, reltol = nothing, reuse_A_if_factorization = false) where {P}
    A !== nothing && (linsolve.A = A)
    b !== nothing && (linsolve.b = b)
    linres = linsolve.A \ linsolve.b
    return FakeLinearSolveJLResult(linsolve, linres)
end

function dolinsolve(precs::P, linsolve; A = nothing, linu = nothing, b = nothing,
        du = nothing, p = nothing, weight = nothing, cachedata = nothing, reltol = nothing,
        reuse_A_if_factorization = false) where {P}
    # Some Algorithms would reuse factorization but it causes the cache to not reset in
    # certain cases
    if A !== nothing
        alg = linsolve.alg
        if (alg isa LinearSolve.AbstractFactorization) ||
           (alg isa LinearSolve.DefaultLinearSolver && !(alg ==
              LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES)))
            # Factorization Algorithm
            !reuse_A_if_factorization && (linsolve.A = A)
        else
            linsolve.A = A
        end
    end
    b !== nothing && (linsolve.b = b)
    linu !== nothing && (linsolve.u = linu)

    Plprev = linsolve.Pl isa ComposePreconditioner ? linsolve.Pl.outer : linsolve.Pl
    Prprev = linsolve.Pr isa ComposePreconditioner ? linsolve.Pr.outer : linsolve.Pr

    _Pl, _Pr = precs(linsolve.A, du, linu, p, nothing, A !== nothing, Plprev, Prprev,
        cachedata)
    if (_Pl !== nothing || _Pr !== nothing)
        _weight = weight === nothing ?
                  (linsolve.Pr isa Diagonal ? linsolve.Pr.diag : linsolve.Pr.inner.diag) :
                  weight
        Pl, Pr = wrapprecs(_Pl, _Pr, _weight)
        linsolve.Pl = Pl
        linsolve.Pr = Pr
    end

    linres = reltol === nothing ? solve!(linsolve) : solve!(linsolve; reltol)

    return linres
end

function wrapprecs(_Pl, _Pr, weight)
    if _Pl !== nothing
        Pl = ComposePreconditioner(InvPreconditioner(Diagonal(_vec(weight))), _Pl)
    else
        Pl = InvPreconditioner(Diagonal(_vec(weight)))
    end

    if _Pr !== nothing
        Pr = ComposePreconditioner(Diagonal(_vec(weight)), _Pr)
    else
        Pr = Diagonal(_vec(weight))
    end

    return Pl, Pr
end

function rfunc(r::R, c2::R, M::R, γ1::R, γ2::R, β::R) where {R <: Real} # R-function for adaptive trust region method
    if (r ≥ c2)
        return (2 * (M - 1 - γ2) * atan(r - c2) + (1 + γ2)) / π
    else
        return (1 - γ1 - β) * (exp(r - c2) + β / (1 - γ1 - β))
    end
end

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

function evaluate_f(f::F, u, p, ::Val{iip}; fu = nothing) where {F, iip <: Bool}
    if iip
        f(fu, u, p)
        return fu
    else
        return f(u, p)
    end
end

function evaluate_f(cache::AbstractNonlinearSolveCache, u, p,
        fu_sym::Val{FUSYM} = Val(nothing)) where {FUSYM}
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

@inline __init_identity_jacobian(u::Number, _) = one(u)
@inline function __init_identity_jacobian(u, fu)
    J = similar(fu, promote_type(eltype(fu), eltype(u)), length(fu), length(u))
    fill!(J, zero(eltype(J)))
    J[diagind(J)] .= one(eltype(J))
    return J
end
@inline function __init_identity_jacobian(u::StaticArray, fu::StaticArray)
    T = promote_type(eltype(fu), eltype(u))
    return MArray{Tuple{prod(Size(fu)), prod(Size(u))}, T}(I)
end
@inline function __init_identity_jacobian(u::SArray, fu::SArray)
    T = promote_type(eltype(fu), eltype(u))
    return SArray{Tuple{prod(Size(fu)), prod(Size(u))}, T}(I)
end

@inline __reinit_identity_jacobian!!(J::Number) = one(J)
@inline function __reinit_identity_jacobian!!(J::AbstractMatrix)
    fill!(J, zero(eltype(J)))
    J[diagind(J)] .= one(eltype(J))
    return J
end
@inline function __reinit_identity_jacobian!!(J::SMatrix)
    S = Size(J)
    return SArray{Tuple{S[1], S[2]}, eltype(J)}(I)
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

# Check Singular Matrix
@inline _issingular(x::Number) = iszero(x)
@inline @generated function _issingular(x::T) where {T}
    hasmethod(issingular, Tuple{T}) && return :(issingular(x))
    return :(__issingular(x))
end
@inline __issingular(x::AbstractMatrix{T}) where {T} = cond(x) > inv(sqrt(eps(real(T))))
@inline __issingular(x) = false ## If SciMLOperator and such

# Safe getproperty
@generated function __getproperty(s::S, ::Val{X}) where {S, X}
    hasfield(S, X) && return :(s.$X)
    return :(nothing)
end

# If factorization is LU then perform that and update the linsolve cache
# else check if the matrix is singular
function __try_factorize_and_check_singular!(linsolve, X)
    if linsolve.cacheval isa LU || linsolve.cacheval isa StaticArrays.LU
        # LU Factorization was used
        linsolve.A = X
        linsolve.cacheval = LinearSolve.do_factorization(linsolve.alg, X, linsolve.b,
            linsolve.u)
        linsolve.isfresh = false

        return !issuccess(linsolve.cacheval), true
    end
    return _issingular(X), false
end
__try_factorize_and_check_singular!(::FakeLinearSolveJLCache, x) = _issingular(x), false

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

# Unalias
@inline __maybe_unaliased(x::Union{Number, SArray}, ::Bool) = x
@inline function __maybe_unaliased(x::AbstractArray, alias::Bool)
    # Spend time coping iff we will mutate the array
    (alias || !can_setindex(typeof(x))) && return x
    return deepcopy(x)
end

# Init ones
@inline function __init_ones(x)
    w = similar(x)
    recursivefill!(w, true)
    return w
end
@inline __init_ones(x::StaticArray) = ones(typeof(x))
