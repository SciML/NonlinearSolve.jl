struct HasAnalyticJacobian end

"""
    __prevfloat_tdir(x, x0, x1)

Move `x` one floating point towards x0.
"""
__prevfloat_tdir(x, x0, x1) = ifelse(x1 > x0, prevfloat(x), nextfloat(x))

"""
    __nextfloat_tdir(x, x0, x1)

Move `x` one floating point towards x1.
"""
__nextfloat_tdir(x, x0, x1) = ifelse(x1 > x0, nextfloat(x), prevfloat(x))

"""
    __max_tdir(a, b, x0, x1)

Return the maximum of `a` and `b` if `x1 > x0`, otherwise return the minimum.
"""
__max_tdir(a, b, x0, x1) = ifelse(x1 > x0, max(a, b), min(a, b))

function __fixed_parameter_function(prob::AbstractNonlinearProblem)
    isinplace(prob) && return @closure (du, u) -> prob.f(du, u, prob.p)
    return Base.Fix2(prob.f, prob.p)
end

function value_and_jacobian(
        ad, prob::AbstractNonlinearProblem, f::F, y, x, cache; J = nothing) where {F}
    x isa Number && return DI.value_and_derivative(f, ad, x, cache)

    if isinplace(prob)
        if cache isa HasAnalyticJacobian
            prob.f.jac(J, x, p)
            f(y, x)
        else
            DI.jacobian!(f, y, J, ad, x, cache)
        end
        return y, J
    else
        cache isa HasAnalyticJacobian && return f(x), prob.f.jac(x, prob.p)
        J === nothing && return DI.value_and_jacobian(f, ad, x, cache)
        y, _ = DI.value_and_jacobian!(f, J, ad, x, cache)
        return y, J
    end
end

function jacobian_cache(ad, prob::AbstractNonlinearProblem, f::F, y, x) where {F}
    x isa Number && return (nothing, DI.prepare_derivative(f, ad, x))

    if isinplace(prob)
        J = __similar(y, length(y), length(x))
        SciMLBase.has_jac(prob.f) && return J, HasAnalyticJacobian()
        return J, DI.prepare_jacobian(f, y, ad, x)
    else
        SciMLBase.has_jac(prob.f) && return nothing, HasAnalyticJacobian()
        J = ArrayInterface.can_setindex(x) ? __similar(y, length(y), length(x)) : nothing
        return J, DI.prepare_jacobian(f, ad, x)
    end
end

function compute_jacobian_and_hessian(
        ad, prob::AbstractNonlinearProblem, f::F, y, x) where {F}
    if x isa Number
        df = @closure x -> DI.derivative(f, ad, x)
        return f(x), df(x), DI.derivative(df, ad, x)
    end

    if isinplace(prob)
        df = @closure x -> begin
            res = __similar(y, promote_type(eltype(y), eltype(x)))
            return DI.jacobian(f, res, ad, x)
        end
        J, H = DI.value_and_jacobian(df, ad, x)
        f(y, x)
        return y, J, H
    end

    df = @closure x -> DI.jacobian(f, ad, x)
    return f(x), df(x), DI.jacobian(df, ad, x)
end

__init_identity_jacobian(u::Number, fu, α = true) = oftype(u, α)
__init_identity_jacobian!!(J::Number) = one(J)
function __init_identity_jacobian(u, fu, α = true)
    J = __similar(u, promote_type(eltype(u), eltype(fu)), length(fu), length(u))
    fill!(J, zero(eltype(J)))
    J[diagind(J)] .= eltype(J)(α)
    return J
end
function __init_identity_jacobian!!(J)
    fill!(J, zero(eltype(J)))
    J[diagind(J)] .= one(eltype(J))
    return J
end
function __init_identity_jacobian!!(J::AbstractVector)
    fill!(J, one(eltype(J)))
    return J
end
function __init_identity_jacobian(u::StaticArray, fu, α = true)
    S1, S2 = length(fu), length(u)
    J = SMatrix{S1, S2, eltype(u)}(I * α)
    return J
end
function __init_identity_jacobian!!(J::SMatrix{S1, S2}) where {S1, S2}
    return SMatrix{S1, S2, eltype(J)}(I)
end
function __init_identity_jacobian!!(J::SVector{S1}) where {S1}
    return ones(SVector{S1, eltype(J)})
end

@inline _vec(v) = vec(v)
@inline _vec(v::Number) = v
@inline _vec(v::AbstractVector) = v

@inline _restructure(y::Number, x::Number) = x
@inline _restructure(y, x) = ArrayInterface.restructure(y, x)

@inline function _get_fx(prob::NonlinearLeastSquaresProblem, x)
    isinplace(prob) &&
        prob.f.resid_prototype === nothing &&
        error("Inplace NonlinearLeastSquaresProblem requires a `resid_prototype`")
    return _get_fx(prob.f, x, prob.p)
end
@inline _get_fx(prob::NonlinearProblem, x) = _get_fx(prob.f, x, prob.p)
@inline function _get_fx(f::NonlinearFunction, x, p)
    if isinplace(f)
        if f.resid_prototype !== nothing
            T = eltype(x)
            return T.(f.resid_prototype)
        else
            fx = __similar(x)
            f(fx, x, p)
            return fx
        end
    else
        return f(x, p)
    end
end

# Termination Conditions Support
# Taken directly from NonlinearSolve.jl
# The default here is different from NonlinearSolve since the userbases are assumed to be
# different. NonlinearSolve is more for robust / cached solvers while SimpleNonlinearSolve
# is meant for low overhead solvers, users can opt into the other termination modes but the
# default is to use the least overhead version.
function init_termination_cache(prob::NonlinearProblem, abstol, reltol, du, u, ::Nothing)
    return init_termination_cache(
        prob, abstol, reltol, du, u, AbsNormTerminationMode(Base.Fix1(maximum, abs)))
end
function init_termination_cache(
        prob::NonlinearLeastSquaresProblem, abstol, reltol, du, u, ::Nothing)
    return init_termination_cache(
        prob, abstol, reltol, du, u, AbsNormTerminationMode(Base.Fix2(norm, 2)))
end

function init_termination_cache(prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem},
        abstol, reltol, du, u, tc::AbstractNonlinearTerminationMode)
    T = promote_type(eltype(du), eltype(u))
    abstol = __get_tolerance(u, abstol, T)
    reltol = __get_tolerance(u, reltol, T)
    tc_ = if hasfield(typeof(tc), :internalnorm) && tc.internalnorm === nothing
        internalnorm = ifelse(
            prob isa NonlinearProblem, Base.Fix1(maximum, abs), Base.Fix2(norm, 2))
        DiffEqBase.set_termination_mode_internalnorm(tc, internalnorm)
    else
        tc
    end
    tc_cache = init(du, u, tc_; abstol, reltol, use_deprecated_retcodes = Val(false))
    return DiffEqBase.get_abstol(tc_cache), DiffEqBase.get_reltol(tc_cache), tc_cache
end

function check_termination(tc_cache, fx, x, xo, prob, alg)
    return check_termination(
        tc_cache, fx, x, xo, prob, alg, DiffEqBase.get_termination_mode(tc_cache))
end
function check_termination(
        tc_cache, fx, x, xo, prob, alg, ::AbstractNonlinearTerminationMode)
    tc_cache(fx, x, xo) &&
        return build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)
    return nothing
end
function check_termination(
        tc_cache, fx, x, xo, prob, alg, ::AbstractSafeNonlinearTerminationMode)
    tc_cache(fx, x, xo) &&
        return build_solution(prob, alg, x, fx; retcode = tc_cache.retcode)
    return nothing
end
function check_termination(
        tc_cache, fx, x, xo, prob, alg, ::AbstractSafeBestNonlinearTerminationMode)
    if tc_cache(fx, x, xo)
        if isinplace(prob)
            prob.f(fx, x, prob.p)
        else
            fx = prob.f(x, prob.p)
        end
        return build_solution(prob, alg, tc_cache.u, fx; retcode = tc_cache.retcode)
    end
    return nothing
end

@inline value(x) = x
@inline value(x::Dual) = ForwardDiff.value(x)
@inline value(x::AbstractArray{<:Dual}) = map(ForwardDiff.value, x)

@inline __eval_f(prob, fx, x) = isinplace(prob) ? (prob.f(fx, x, prob.p); fx) :
                                prob.f(x, prob.p)

# Unalias
@inline __maybe_unaliased(x::Union{Number, SArray}, ::Bool) = x
@inline function __maybe_unaliased(x::AbstractArray, alias::Bool)
    # Spend time coping iff we will mutate the array
    (alias || !ArrayInterface.can_setindex(typeof(x))) && return x
    return deepcopy(x)
end

# Decide which AD backend to use
@inline __get_concrete_autodiff(prob, ad::AbstractADType; kwargs...) = ad
@inline function __get_concrete_autodiff(prob, ad::AutoForwardDiff{nothing}; kwargs...)
    return AutoForwardDiff(; chunksize = ForwardDiff.pickchunksize(length(prob.u0)), ad.tag)
end
@inline function __get_concrete_autodiff(
        prob, ad::AutoPolyesterForwardDiff{nothing}; kwargs...)
    return AutoPolyesterForwardDiff(;
        chunksize = ForwardDiff.pickchunksize(length(prob.u0)), ad.tag)
end
@inline function __get_concrete_autodiff(prob, ::Nothing; kwargs...)
    return ifelse(ForwardDiff.can_dual(eltype(prob.u0)),
        AutoForwardDiff(; chunksize = ForwardDiff.pickchunksize(length(prob.u0))),
        AutoFiniteDiff())
end

@inline __reshape(x::Number, args...) = x
@inline __reshape(x::AbstractArray, args...) = reshape(x, args...)

# Override cases which might be used in a kernel launch
__get_tolerance(x, η, ::Type{T}) where {T} = DiffEqBase._get_tolerance(η, T)
function __get_tolerance(x::Union{SArray, Number}, ::Nothing, ::Type{T}) where {T}
    η = real(oneunit(T)) * (eps(real(one(T))))^(real(T)(0.8))
    return T(η)
end

# Extension
function __zygote_compute_nlls_vjp end

function __similar(x, args...; kwargs...)
    y = similar(x, args...; kwargs...)
    return __init_bigfloat_array!!(y)
end

function __init_bigfloat_array!!(x)
    if ArrayInterface.can_setindex(x)
        eltype(x) <: BigFloat && fill!(x, BigFloat(0))
        return x
    end
    return x
end
