struct SimpleNonlinearSolveTag end

function ForwardDiff.checktag(::Type{<:ForwardDiff.Tag{<:SimpleNonlinearSolveTag, <:T}},
        f::F, x::AbstractArray{T}) where {T, F}
    return true
end

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

__standard_tag(::Nothing, x) = ForwardDiff.Tag(SimpleNonlinearSolveTag(), eltype(x))
__standard_tag(tag::ForwardDiff.Tag, _) = tag
__standard_tag(tag, x) = ForwardDiff.Tag(tag, eltype(x))

__pick_forwarddiff_chunk(x) = ForwardDiff.Chunk(length(x))
function __pick_forwarddiff_chunk(x::StaticArray)
    L = prod(Size(x))
    if L ≤ ForwardDiff.DEFAULT_CHUNK_THRESHOLD
        return ForwardDiff.Chunk{L}()
    else
        return ForwardDiff.Chunk{ForwardDiff.DEFAULT_CHUNK_THRESHOLD}()
    end
end

function __get_jacobian_config(ad::AutoForwardDiff{CS}, f, x) where {CS}
    ck = (CS === nothing || CS ≤ 0) ? __pick_forwarddiff_chunk(x) : ForwardDiff.Chunk{CS}()
    tag = __standard_tag(ad.tag, x)
    return ForwardDiff.JacobianConfig(f, x, ck, tag)
end
function __get_jacobian_config(ad::AutoForwardDiff{CS}, f!, y, x) where {CS}
    ck = (CS === nothing || CS ≤ 0) ? __pick_forwarddiff_chunk(x) : ForwardDiff.Chunk{CS}()
    tag = __standard_tag(ad.tag, x)
    return ForwardDiff.JacobianConfig(f!, y, x, ck, tag)
end

function __get_jacobian_config(ad::AutoPolyesterForwardDiff{CS}, args...) where {CS}
    x = last(args)
    return (CS === nothing || CS ≤ 0) ? __pick_forwarddiff_chunk(x) :
           ForwardDiff.Chunk{CS}()
end

"""
    value_and_jacobian(ad, f, y, x, p, cache; J = nothing)

Compute `f(x), d/dx f(x)` in the most efficient way based on `ad`. None of the arguments
except `cache` (& `J` if not nothing) are mutated.
"""
function value_and_jacobian(ad, f::F, y, x::X, p, cache; J = nothing) where {F, X}
    if isinplace(f)
        _f = (du, u) -> f(du, u, p)
        if DiffEqBase.has_jac(f)
            f.jac(J, x, p)
            _f(y, x)
            return y, J
        elseif ad isa AutoForwardDiff
            res = DiffResults.DiffResult(y, J)
            ForwardDiff.jacobian!(res, _f, y, x, cache)
            return DiffResults.value(res), DiffResults.jacobian(res)
        elseif ad isa AutoFiniteDiff
            FiniteDiff.finite_difference_jacobian!(J, _f, x, cache)
            _f(y, x)
            return y, J
        elseif ad isa AutoPolyesterForwardDiff
            __polyester_forwarddiff_jacobian!(_f, y, J, x, cache)
            return y, J
        else
            throw(ArgumentError("Unsupported AD method: $(ad)"))
        end
    else
        _f = Base.Fix2(f, p)
        if DiffEqBase.has_jac(f)
            return _f(x), f.jac(x, p)
        elseif ad isa AutoForwardDiff
            if ArrayInterface.can_setindex(x)
                res = DiffResults.DiffResult(y, J)
                ForwardDiff.jacobian!(res, _f, x, cache)
                return DiffResults.value(res), DiffResults.jacobian(res)
            else
                J_fd = ForwardDiff.jacobian(_f, x, cache)
                return _f(x), J_fd
            end
        elseif ad isa AutoFiniteDiff
            J_fd = FiniteDiff.finite_difference_jacobian(_f, x, cache)
            return _f(x), J_fd
        elseif ad isa AutoPolyesterForwardDiff
            __polyester_forwarddiff_jacobian!(_f, J, x, cache)
            return _f(x), J
        else
            throw(ArgumentError("Unsupported AD method: $(ad)"))
        end
    end
end

# Declare functions
function __polyester_forwarddiff_jacobian! end

function value_and_jacobian(ad, f::F, y, x::Number, p, cache; J = nothing) where {F}
    if DiffEqBase.has_jac(f)
        return f(x, p), f.jac(x, p)
    elseif ad isa AutoForwardDiff
        T = typeof(__standard_tag(ad.tag, x))
        out = f(ForwardDiff.Dual{T}(x, one(x)), p)
        return ForwardDiff.value(out), ForwardDiff.extract_derivative(T, out)
    elseif ad isa AutoPolyesterForwardDiff
        # Just use ForwardDiff
        T = typeof(__standard_tag(nothing, x))
        out = f(ForwardDiff.Dual{T}(x, one(x)), p)
        return ForwardDiff.value(out), ForwardDiff.extract_derivative(T, out)
    elseif ad isa AutoFiniteDiff
        _f = Base.Fix2(f, p)
        return _f(x), FiniteDiff.finite_difference_derivative(_f, x, ad.fdtype)
    else
        throw(ArgumentError("Unsupported AD method: $(ad)"))
    end
end

"""
    jacobian_cache(ad, f, y, x, p) --> J, cache

Returns a Jacobian Matrix and a cache for the Jacobian computation.
"""
function jacobian_cache(ad, f::F, y, x::X, p) where {F, X <: AbstractArray}
    if isinplace(f)
        _f = (du, u) -> f(du, u, p)
        J = similar(y, length(y), length(x))
        if DiffEqBase.has_jac(f)
            return J, nothing
        elseif ad isa AutoForwardDiff || ad isa AutoPolyesterForwardDiff
            return J, __get_jacobian_config(ad, _f, y, x)
        elseif ad isa AutoFiniteDiff
            return J, FiniteDiff.JacobianCache(copy(x), copy(y), copy(y), ad.fdtype)
        else
            throw(ArgumentError("Unsupported AD method: $(ad)"))
        end
    else
        _f = Base.Fix2(f, p)
        if DiffEqBase.has_jac(f)
            return nothing, nothing
        elseif ad isa AutoForwardDiff
            J = ArrayInterface.can_setindex(x) ? similar(y, length(y), length(x)) : nothing
            return J, __get_jacobian_config(ad, _f, x)
        elseif ad isa AutoPolyesterForwardDiff
            @assert ArrayInterface.can_setindex(x) "PolyesterForwardDiff requires mutable inputs. Use AutoForwardDiff instead."
            J = similar(y, length(y), length(x))
            return J, __get_jacobian_config(ad, _f, x)
        elseif ad isa AutoFiniteDiff
            return nothing, FiniteDiff.JacobianCache(copy(x), copy(y), copy(y), ad.fdtype)
        else
            throw(ArgumentError("Unsupported AD method: $(ad)"))
        end
    end
end

jacobian_cache(ad, f::F, y, x::Number, p) where {F} = nothing, nothing

function compute_jacobian_and_hessian(ad::AutoForwardDiff, prob, _, x::Number)
    fx = prob.f(x, prob.p)
    J_fn = Base.Fix1(ForwardDiff.derivative, Base.Fix2(prob.f, prob.p))
    dfx = J_fn(x)
    d2fx = ForwardDiff.derivative(J_fn, x)
    return fx, dfx, d2fx
end

function compute_jacobian_and_hessian(ad::AutoForwardDiff, prob, fx, x)
    if isinplace(prob)
        error("Inplace version for Nested ForwardDiff Not Implemented Yet!")
    else
        f = Base.Fix2(prob.f, prob.p)
        fx = f(x)
        J_fn = Base.Fix1(ForwardDiff.jacobian, f)
        dfx = J_fn(x)
        d2fx = ForwardDiff.jacobian(J_fn, x)
        return fx, dfx, d2fx
    end
end

function compute_jacobian_and_hessian(ad::AutoFiniteDiff, prob, _, x::Number)
    fx = prob.f(x, prob.p)
    J_fn = x -> FiniteDiff.finite_difference_derivative(Base.Fix2(prob.f, prob.p), x,
        ad.fdtype)
    dfx = J_fn(x)
    d2fx = FiniteDiff.finite_difference_derivative(J_fn, x, ad.fdtype)
    return fx, dfx, d2fx
end

function compute_jacobian_and_hessian(ad::AutoFiniteDiff, prob, fx, x)
    if isinplace(prob)
        error("Inplace version for Nested FiniteDiff Not Implemented Yet!")
    else
        f = Base.Fix2(prob.f, prob.p)
        fx = f(x)
        J_fn = x -> FiniteDiff.finite_difference_jacobian(f, x, ad.fdtype)
        dfx = J_fn(x)
        d2fx = FiniteDiff.finite_difference_jacobian(J_fn, x, ad.fdtype)
        return fx, dfx, d2fx
    end
end

__init_identity_jacobian(u::Number, _) = one(u)
__init_identity_jacobian!!(J::Number) = one(J)
function __init_identity_jacobian(u, fu)
    J = similar(u, promote_type(eltype(u), eltype(fu)), length(fu), length(u))
    fill!(J, zero(eltype(J)))
    J[diagind(J)] .= one(eltype(J))
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
function __init_identity_jacobian(u::StaticArray, fu)
    S1, S2 = length(fu), length(u)
    J = SMatrix{S1, S2, eltype(u)}(I)
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
    isinplace(prob) && prob.f.resid_prototype === nothing &&
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
            fx = similar(x)
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
function init_termination_cache(abstol, reltol, du, u, ::Nothing)
    return init_termination_cache(abstol, reltol, du, u, AbsNormTerminationMode())
end
function init_termination_cache(abstol, reltol, du, u, tc::AbstractNonlinearTerminationMode)
    T = promote_type(eltype(du), eltype(u))
    abstol !== nothing && (abstol = T(abstol))
    reltol !== nothing && (reltol = T(reltol))
    tc_cache = init(du, u, tc; abstol, reltol)
    return DiffEqBase.get_abstol(tc_cache), DiffEqBase.get_reltol(tc_cache), tc_cache
end

function check_termination(tc_cache, fx, x, xo, prob, alg)
    return check_termination(tc_cache, fx, x, xo, prob, alg,
        DiffEqBase.get_termination_mode(tc_cache))
end
function check_termination(tc_cache, fx, x, xo, prob, alg,
        ::AbstractNonlinearTerminationMode)
    if Bool(tc_cache(fx, x, xo))
        return build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)
    end
    return nothing
end
function check_termination(tc_cache, fx, x, xo, prob, alg,
        ::AbstractSafeNonlinearTerminationMode)
    if Bool(tc_cache(fx, x, xo))
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.Success
            retcode = ReturnCode.Success
        elseif tc_cache.retcode == NonlinearSafeTerminationReturnCode.PatienceTermination
            retcode = ReturnCode.ConvergenceFailure
        elseif tc_cache.retcode == NonlinearSafeTerminationReturnCode.ProtectiveTermination
            retcode = ReturnCode.Unstable
        else
            error("Unknown termination code: $(tc_cache.retcode)")
        end
        return build_solution(prob, alg, x, fx; retcode)
    end
    return nothing
end
function check_termination(tc_cache, fx, x, xo, prob, alg,
        ::AbstractSafeBestNonlinearTerminationMode)
    if Bool(tc_cache(fx, x, xo))
        if tc_cache.retcode == NonlinearSafeTerminationReturnCode.Success
            retcode = ReturnCode.Success
        elseif tc_cache.retcode == NonlinearSafeTerminationReturnCode.PatienceTermination
            retcode = ReturnCode.ConvergenceFailure
        elseif tc_cache.retcode == NonlinearSafeTerminationReturnCode.ProtectiveTermination
            retcode = ReturnCode.Unstable
        else
            error("Unknown termination code: $(tc_cache.retcode)")
        end
        if isinplace(prob)
            prob.f(fx, x, prob.p)
        else
            fx = prob.f(x, prob.p)
        end
        return build_solution(prob, alg, tc_cache.u, fx; retcode)
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
@inline __get_concrete_autodiff(prob, ad::ADTypes.AbstractADType; kwargs...) = ad
@inline function __get_concrete_autodiff(prob, ::Nothing; polyester::Val{P} = Val(true),
        kwargs...) where {P}
    if ForwardDiff.can_dual(eltype(prob.u0))
        if P && __is_extension_loaded(Val(:PolyesterForwardDiff)) &&
           !(prob.u0 isa Number) && ArrayInterface.can_setindex(prob.u0)
            return AutoPolyesterForwardDiff()
        else
            return AutoForwardDiff()
        end
    else
        return AutoFiniteDiff()
    end
end

@inline __reshape(x::Number, args...) = x
@inline __reshape(x::AbstractArray, args...) = reshape(x, args...)

# Override cases which might be used in a kernel launch
__get_tolerance(x, η, ::Type{T}) where {T} = DiffEqBase._get_tolerance(η, T)
function __get_tolerance(x::Union{SArray, Number}, ::Nothing, ::Type{T}) where {T}
    η = real(oneunit(T)) * (eps(real(one(T))))^(real(T)(0.8))
    return T(η)
end
