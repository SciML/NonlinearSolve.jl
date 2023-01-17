
@inline UNITLESS_ABS2(x) = real(abs2(x))
@inline DEFAULT_NORM(u::Union{AbstractFloat, Complex}) = @fastmath abs(u)
@inline function DEFAULT_NORM(u::Array{T}) where {T <: Union{AbstractFloat, Complex}}
    sqrt(real(sum(abs2, u)) / length(u))
end
@inline function DEFAULT_NORM(u::StaticArraysCore.StaticArray{T}) where {
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

function alg_difftype(alg::AbstractNewtonAlgorithm{CS, AD, FDT, ST, CJ}) where {CS, AD, FDT,
                                                                                ST, CJ}
    FDT
end

function concrete_jac(alg::AbstractNewtonAlgorithm{CS, AD, FDT, ST, CJ}) where {CS, AD, FDT,
                                                                                ST, CJ}
    CJ
end

function get_chunksize(alg::AbstractNewtonAlgorithm{CS, AD, FDT, ST, CJ}) where {CS, AD,
                                                                                 FDT,
                                                                                 ST, CJ}
    Val(CS)
end

function standardtag(alg::AbstractNewtonAlgorithm{CS, AD, FDT, ST, CJ}) where {CS, AD, FDT,
                                                                               ST, CJ}
    ST
end

DEFAULT_PRECS(W, du, u, p, t, newW, Plprev, Prprev, cachedata) = nothing, nothing

function dolinsolve(precs::P, linsolve; A = nothing, linu = nothing, b = nothing,
                    du = nothing, u = nothing, p = nothing, t = nothing,
                    weight = nothing, cachedata = nothing,
                    reltol = nothing) where {P}
    A !== nothing && (linsolve = LinearSolve.set_A(linsolve, A))
    b !== nothing && (linsolve = LinearSolve.set_b(linsolve, b))
    linu !== nothing && (linsolve = LinearSolve.set_u(linsolve, linu))

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
        linsolve = LinearSolve.set_prec(linsolve, Pl, Pr)
    end

    linres = if reltol === nothing
        solve(linsolve)
    else
        solve(linsolve; reltol)
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

function dogleg!(cache)
    @unpack g, H, trust_r = cache
    # Compute the Newton step.
    δN = -H \ g
    # Test if the full step is within the trust region.
    if norm(δN) ≤ trust_r
        cache.step_size = δN
        return
    end

    # Calcualte Cauchy point, optimum along the steepest descent direction.
    δsd = -g
    norm_δsd = norm(δsd)
    if norm_δsd ≥ trust_r
        cache.step_size = δsd .* trust_r / norm_δsd
        return
    end

    # Find the intersection point on the boundary.
    N_sd = δN - δsd
    dot_N_sd = dot(N_sd, N_sd)
    dot_sd_N_sd = dot(δsd, N_sd)
    dot_sd = dot(δsd, δsd)
    fact = dot_sd_N_sd^2 - dot_N_sd * (dot_sd - trust_r^2)
    τ = (-dot_sd_N_sd + sqrt(fact)) / dot_N_sd
    cache.step_size = δsd + τ * N_sd
end
