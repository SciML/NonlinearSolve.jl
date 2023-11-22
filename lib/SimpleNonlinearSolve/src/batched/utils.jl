macro maybeinplace(iip::Symbol, expr::Expr, u0::Union{Symbol, Nothing} = nothing)
    @assert expr.head == :(=)
    x1, x2 = expr.args
    @assert x2.head == :call
    f, x... = x2.args
    define_expr = u0 === nothing ? :() : :($(x1) = similar($(u0)))
    return quote
        if $(esc(iip))
            $(esc(define_expr))
            $(esc(f))($(esc(x1)), $(esc.(x)...))
        else
            $(esc(expr))
        end
    end
end

function _get_tolerance(η, tc_η, ::Type{T}) where {T}
    fallback_η = real(oneunit(T)) * (eps(real(one(T))))^(4 // 5)
    return ifelse(η !== nothing, η, ifelse(tc_η !== nothing, tc_η, fallback_η))
end

function _construct_batched_problem_structure(prob)
    return _construct_batched_problem_structure(prob.u0,
        prob.f,
        prob.p,
        Val(SciMLBase.isinplace(prob)))
end

function _construct_batched_problem_structure(u0::AbstractArray{T, N},
        f,
        p,
        ::Val{iip}) where {T, N, iip}
    # Reconstruct `u`
    reconstruct = N == 2 ? identity : Base.Fix2(reshape, size(u0))
    # Standardize `u`
    standardize = N == 2 ? identity :
                  (N == 1 ? Base.Fix2(reshape, (:, 1)) :
                   Base.Fix2(reshape, (:, size(u0, ndims(u0)))))
    # Updated Function
    f_modified = if iip
        function f_modified_iip(du, u)
            f(reconstruct(du), reconstruct(u), p)
            return standardize(du)
        end
    else
        f_modified_oop(u) = standardize(f(reconstruct(u), p))
    end
    return standardize(u0), f_modified, reconstruct
end

@views function _init_𝓙(x::AbstractMatrix)
    𝓙 = ArrayInterface.zeromatrix(x[:, 1])
    if ismutable(x)
        𝓙[diagind(𝓙)] .= one(eltype(x))
    else
        𝓙 .+= I
    end
    return repeat(𝓙, 1, 1, size(x, 2))
end

_result_from_storage(::Nothing, xₙ, fₙ, args...) = ReturnCode.Success, xₙ, fₙ
function _result_from_storage(storage::NLSolveSafeTerminationResult, xₙ, fₙ, f, mode, iip)
    if storage.return_code == DiffEqBase.NLSolveSafeTerminationReturnCode.Success
        return ReturnCode.Success, xₙ, fₙ
    else
        if mode ∈ DiffEqBase.SAFE_BEST_TERMINATION_MODES
            @maybeinplace iip fₙ=f(xₙ)
            return ReturnCode.Terminated, storage.u, fₙ
        else
            return ReturnCode.Terminated, xₙ, fₙ
        end
    end
end

function _get_storage(mode, u)
    return mode ∈ DiffEqBase.SAFE_TERMINATION_MODES ?
           NLSolveSafeTerminationResult(mode ∈ DiffEqBase.SAFE_BEST_TERMINATION_MODES ? u :
                                        nothing) : nothing
end
