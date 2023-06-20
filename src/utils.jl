
@inline UNITLESS_ABS2(x) = real(abs2(x))
@inline DEFAULT_NORM(u::Union{AbstractFloat, Complex}) = @fastmath abs(u)
@inline function DEFAULT_NORM(u::Array{T}) where {T <: Union{AbstractFloat, Complex}}
    sqrt(real(sum(abs2, u)) / length(u))
end
@inline function DEFAULT_NORM(u::StaticArraysCore.StaticArray{
    T,
}) where {
    T <: Union{
        AbstractFloat,
        Complex}}
    sqrt(real(sum(abs2, u)) / length(u))
end
@inline function DEFAULT_NORM(u::RecursiveArrayTools.AbstractVectorOfArray)
    sum(sqrt(real(sum(UNITLESS_ABS2, _u)) / length(_u)) for _u in u.u)
end
@inline DEFAULT_NORM(u::AbstractArray) = sqrt(real(sum(UNITLESS_ABS2, u)) / length(u))
@inline DEFAULT_NORM(u) = norm(u)

alg_autodiff(alg::AbstractNewtonAlgorithm{CS, AD}) where {CS, AD} = AD
alg_autodiff(alg) = false

"""
value_derivative(f, x)

Compute `f(x), d/dx f(x)` in the most efficient way.
"""
function value_derivative(f::F, x::R) where {F, R}
    T = typeof(ForwardDiff.Tag(f, R))
    out = f(ForwardDiff.Dual{T}(x, one(x)))
    ForwardDiff.value(out), ForwardDiff.extract_derivative(T, out)
end

# Todo: improve this dispatch
function value_derivative(f::F, x::StaticArraysCore.SVector) where {F}
    f(x), ForwardDiff.jacobian(f, x)
end

value(x) = x
value(x::Dual) = ForwardDiff.value(x)
value(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)

_vec(v) = vec(v)
_vec(v::Number) = v
_vec(v::AbstractVector) = v

function alg_difftype(alg::AbstractNewtonAlgorithm{
    CS,
    AD,
    FDT,
    ST,
    CJ,
}) where {CS, AD, FDT,
    ST, CJ}
    FDT
end

function concrete_jac(alg::AbstractNewtonAlgorithm{
    CS,
    AD,
    FDT,
    ST,
    CJ,
}) where {CS, AD, FDT,
    ST, CJ}
    CJ
end

function get_chunksize(alg::AbstractNewtonAlgorithm{
    CS,
    AD,
    FDT,
    ST,
    CJ,
}) where {CS, AD,
    FDT,
    ST, CJ}
    Val(CS)
end

function standardtag(alg::AbstractNewtonAlgorithm{
    CS,
    AD,
    FDT,
    ST,
    CJ,
}) where {CS, AD, FDT,
    ST, CJ}
    ST
end

DEFAULT_PRECS(W, du, u, p, t, newW, Plprev, Prprev, cachedata) = nothing, nothing

function dolinsolve(precs::P, linsolve; A = nothing, linu = nothing, b = nothing,
    du = nothing, u = nothing, p = nothing, t = nothing,
    weight = nothing, cachedata = nothing,
    reltol = nothing) where {P}
    A !== nothing && (linsolve.A = A)
    b !== nothing && (linsolve.b = b)
    linu !== nothing && (linsolve.u = linu)

    Plprev = linsolve.Pl isa LinearSolve.ComposePreconditioner ? linsolve.Pl.outer :
             linsolve.Pl
    Prprev = linsolve.Pr isa LinearSolve.ComposePreconditioner ? linsolve.Pr.outer :
             linsolve.Pr

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

    linres = if reltol === nothing
        solve!(linsolve)
    else
        solve!(linsolve; reltol)
    end

    return linres
end

function wrapprecs(_Pl, _Pr, weight)
    if _Pl !== nothing
        Pl = LinearSolve.ComposePreconditioner(LinearSolve.InvPreconditioner(Diagonal(_vec(weight))),
            _Pl)
    else
        Pl = LinearSolve.InvPreconditioner(Diagonal(_vec(weight)))
    end

    if _Pr !== nothing
        Pr = LinearSolve.ComposePreconditioner(Diagonal(_vec(weight)), _Pr)
    else
        Pr = Diagonal(_vec(weight))
    end
    Pl, Pr
end

function _nfcount(N, ::Type{diff_type}) where {diff_type}
    if diff_type === Val{:complex}
        tmp = N
    elseif diff_type === Val{:forward}
        tmp = N + 1
    else
        tmp = 2N
    end
    tmp
end

function get_loss(fu)
    return norm(fu)^2 / 2
end

function rfunc(r::R, c2::R, M::R, γ1::R, γ2::R, β::R) where {R <: Real} # R-function for adaptive trust region method
    if (r >= c2)
        return (2 * (M - 1 - γ2) * atan(r - c2) + (1 + γ2)) / π
    else
        return (1 - γ1 - β) * (exp(r - c2) + β / (1 - γ1 - β))
    end
end
