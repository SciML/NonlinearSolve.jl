
@inline UNITLESS_ABS2(x) = real(abs2(x))
@inline DEFAULT_NORM(u::Union{AbstractFloat, Complex}) = @fastmath abs(u)
@inline function DEFAULT_NORM(u::Array{T}) where {T <: Union{AbstractFloat, Complex}}
    return sqrt(real(sum(abs2, u)) / length(u))
end
@inline function DEFAULT_NORM(u::StaticArray{<:Union{AbstractFloat, Complex}})
    return sqrt(real(sum(abs2, u)) / length(u))
end
@inline function DEFAULT_NORM(u::AbstractVectorOfArray)
    return sum(sqrt(real(sum(UNITLESS_ABS2, _u)) / length(_u)) for _u in u.u)
end
@inline DEFAULT_NORM(u::AbstractArray) = sqrt(real(sum(UNITLESS_ABS2, u)) / length(u))
@inline DEFAULT_NORM(u) = norm(u)

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
    tag = _unwrap_val(standardtag)
    ad && return AutoForwardDiff{_unwrap_val(chunk_size), typeof(tag)}(tag)
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

function value_derivative(f::F, x::SVector) where {F}
    f(x), ForwardDiff.jacobian(f, x)
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

function dolinsolve(precs::P, linsolve; A = nothing, linu = nothing, b = nothing,
    du = nothing, u = nothing, p = nothing, t = nothing, weight = nothing,
    cachedata = nothing, reltol = nothing) where {P}
    A !== nothing && (linsolve.A = A)
    b !== nothing && (linsolve.b = b)
    linu !== nothing && (linsolve.u = linu)

    Plprev = linsolve.Pl isa ComposePreconditioner ? linsolve.Pl.outer : linsolve.Pl
    Prprev = linsolve.Pr isa ComposePreconditioner ? linsolve.Pr.outer : linsolve.Pr

    _Pl, _Pr = precs(linsolve.A, du, u, p, nothing, A !== nothing, Plprev, Prprev,
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

function _nfcount(N, ::Type{diff_type}) where {diff_type}
    if diff_type === Val{:complex}
        tmp = N
    elseif diff_type === Val{:forward}
        tmp = N + 1
    else
        tmp = 2N
    end
    return tmp
end

get_loss(fu) = norm(fu)^2 / 2

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

_maybe_mutable(x, ::AbstractFiniteDifferencesMode) = _mutable(x)
# The shadow allocated for Enzyme needs to be mutable
_maybe_mutable(x, ::AutoSparseEnzyme) = _mutable(x)
_maybe_mutable(x, _) = x

# Helper function to get value of `f(u, p)`
function evaluate_f(prob::Union{NonlinearProblem{uType, iip},
        NonlinearLeastSquaresProblem{uType, iip}}, u) where {uType, iip}
    @unpack f, u0, p = prob
    if iip
        fu = f.resid_prototype === nothing ? zero(u) : f.resid_prototype
        f(fu, u, p)
    else
        fu = _mutable(f(u, p))
    end
    return fu
end

evaluate_f(cache, u; fu = nothing) = evaluate_f(cache.f, u, cache.p, Val(cache.iip); fu)

function evaluate_f(f, u, p, ::Val{iip}; fu = nothing) where {iip}
    if iip
        f(fu, u, p)
        return fu
    else
        return f(u, p)
    end
end

"""
    __matmul!(C, A, B)

Defaults to `mul!(C, A, B)`. However, for sparse matrices uses `C .= A * B`.
"""
__matmul!(C, A, B) = mul!(C, A, B)
__matmul!(C::AbstractSparseMatrix, A, B) = C .= A * B

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
    ad = if eltype(prob.u0) <: Complex
        # Use Finite Differencing
        use_sparse_ad ? AutoSparseFiniteDiff() : AutoFiniteDiff()
    else
        use_sparse_ad ? AutoSparseForwardDiff() : AutoForwardDiff{nothing, Nothing}(nothing)
    end
    return set_ad(alg, ad)
end

__init_identity_jacobian(u::Number, _) = u
function __init_identity_jacobian(u, fu)
    return convert(parameterless_type(_mutable(u)),
        Matrix{eltype(u)}(I, length(fu), length(u)))
end
function __init_identity_jacobian(u::StaticArray, fu)
    return convert(MArray{Tuple{length(fu), length(u)}},
        Matrix{eltype(u)}(I, length(fu), length(u)))
end

function __init_low_rank_jacobian(u::StaticArray, fu, threshold::Int)
    Vᵀ = convert(MArray{Tuple{length(u), threshold}},
        zeros(eltype(u), length(u), threshold))
    U = convert(MArray{Tuple{threshold, length(u)}}, zeros(eltype(u), threshold, length(u)))
    return U, Vᵀ
end
function __init_low_rank_jacobian(u, fu, threshold::Int)
    Vᵀ = convert(parameterless_type(_mutable(u)), zeros(eltype(u), length(u), threshold))
    U = convert(parameterless_type(_mutable(u)), zeros(eltype(u), threshold, length(u)))
    return U, Vᵀ
end

# Check Singular Matrix
_issingular(x::Number) = iszero(x)
@generated function _issingular(x::T) where {T}
    hasmethod(issingular, Tuple{T}) && return :(issingular(x))
    return :(__issingular(x))
end
__issingular(x::AbstractMatrix{T}) where {T} = cond(x) > inv(sqrt(eps(T)))
__issingular(x) = false ## If SciMLOperator and such

# If factorization is LU then perform that and update the linsolve cache
# else check if the matrix is singular
function _try_factorize_and_check_singular!(linsolve, X)
    if linsolve.cacheval isa LU
        # LU Factorization was used
        linsolve.A = X
        linsolve.cacheval = LinearSolve.do_factorization(linsolve.alg, X, linsolve.b,
            linsolve.u)
        linsolve.isfresh = false

        return !issuccess(linsolve.cacheval), true
    end
    return _issingular(X), false
end
_try_factorize_and_check_singular!(::Nothing, x) = _issingular(x), false

@inline _reshape(x, args...) = reshape(x, args...)
@inline _reshape(x::Number, args...) = x

@generated function _axpy!(α, x, y)
    hasmethod(axpy!, Tuple{α, x, y}) && return :(axpy!(α, x, y))
    return :(@. y += α * x)
end

@inline _needs_square_A(_, ::Number) = true
@inline _needs_square_A(_, ::StaticArray) = true
@inline _needs_square_A(alg, _) = LinearSolve.needs_square_A(alg.linsolve)

# Define special concatenation for certain Array combinations
@inline _vcat(x, y) = vcat(x, y)
