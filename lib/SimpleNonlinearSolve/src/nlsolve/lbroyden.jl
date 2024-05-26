"""
    SimpleLimitedMemoryBroyden(; threshold::Union{Val, Int} = Val(27),
        linesearch = Val(false), alpha = nothing)

A limited memory implementation of Broyden. This method applies the L-BFGS scheme to
Broyden's method.

If the threshold is larger than the problem size, then this method will use `SimpleBroyden`.

### Keyword Arguments:

  - `linesearch`: If `linesearch` is `Val(true)`, then we use the `LiFukushimaLineSearch`
    [li2000derivative](@cite) line search else no line search is used. For advanced
    customization of the line search, use the `LimitedMemoryBroyden` algorithm in
    `NonlinearSolve.jl`.
  - `alpha`: Scale the initial jacobian initialization with `alpha`. If it is `nothing`, we
    will compute the scaling using `2 * norm(fu) / max(norm(u), true)`.
"""
@concrete struct SimpleLimitedMemoryBroyden{threshold, linesearch} <:
                 AbstractSimpleNonlinearSolveAlgorithm
    alpha
end

__get_threshold(::SimpleLimitedMemoryBroyden{threshold}) where {threshold} = Val(threshold)
__use_linesearch(::SimpleLimitedMemoryBroyden{Th, LS}) where {Th, LS} = Val(LS)

function SimpleLimitedMemoryBroyden(;
        threshold::Union{Val, Int} = Val(27), linesearch = Val(false), alpha = nothing)
    return SimpleLimitedMemoryBroyden{_unwrap_val(threshold), _unwrap_val(linesearch)}(alpha)
end

function SciMLBase.__solve(prob::NonlinearProblem, alg::SimpleLimitedMemoryBroyden,
        args...; termination_condition = nothing, kwargs...)
    if prob.u0 isa SArray
        if termination_condition === nothing ||
           termination_condition isa AbsNormTerminationMode
            return __static_solve(prob, alg, args...; termination_condition, kwargs...)
        end
        @warn "Specifying `termination_condition = $(termination_condition)` for \
               `SimpleLimitedMemoryBroyden` with `SArray` is not non-allocating. Use \
               either `termination_condition = AbsNormTerminationMode()` or \
               `termination_condition = nothing`." maxlog=1
    end
    return __generic_solve(prob, alg, args...; termination_condition, kwargs...)
end

@views function __generic_solve(prob::NonlinearProblem, alg::SimpleLimitedMemoryBroyden,
        args...; abstol = nothing, reltol = nothing, maxiters = 1000,
        alias_u0 = false, termination_condition = nothing, kwargs...)
    x = __maybe_unaliased(prob.u0, alias_u0)
    threshold = __get_threshold(alg)
    η = min(_unwrap_val(threshold), maxiters)

    # For scalar problems / if the threshold is larger than problem size just use Broyden
    if x isa Number || length(x) ≤ η
        return SciMLBase.__solve(
            prob, SimpleBroyden(; linesearch = __use_linesearch(alg)), args...;
            abstol, reltol, maxiters, termination_condition, kwargs...)
    end

    fx = _get_fx(prob, x)

    U, Vᵀ = __init_low_rank_jacobian(x, fx, x isa StaticArray ? threshold : Val(η))

    abstol, reltol, tc_cache = init_termination_cache(
        prob, abstol, reltol, fx, x, termination_condition)

    @bb xo = copy(x)
    @bb δx = copy(fx)
    @bb δx .*= -1
    @bb fo = copy(fx)
    @bb δf = copy(fx)

    @bb vᵀ_cache = copy(x)
    Tcache = __lbroyden_threshold_cache(x, x isa StaticArray ? threshold : Val(η))
    @bb mat_cache = copy(x)

    ls_cache = __use_linesearch(alg) === Val(true) ? LiFukushimaLineSearch()(prob, fx, x) :
               nothing

    for i in 1:maxiters
        α = ls_cache === nothing ? true : ls_cache(x, δx)
        @bb @. x = xo + α * δx
        fx = __eval_f(prob, fx, x)
        @bb @. δf = fx - fo

        # Termination Checks
        tc_sol = check_termination(tc_cache, fx, x, xo, prob, alg)
        tc_sol !== nothing && return tc_sol

        _U = selectdim(U, 2, 1:min(η, i - 1))
        _Vᵀ = selectdim(Vᵀ, 1, 1:min(η, i - 1))

        vᵀ = _rmatvec!!(vᵀ_cache, Tcache, _U, _Vᵀ, δx)
        mvec = _matvec!!(mat_cache, Tcache, _U, _Vᵀ, δf)
        d = dot(vᵀ, δf)
        @bb @. δx = (δx - mvec) / d

        selectdim(U, 2, mod1(i, η)) .= _vec(δx)
        selectdim(Vᵀ, 1, mod1(i, η)) .= _vec(vᵀ)

        _U = selectdim(U, 2, 1:min(η, i))
        _Vᵀ = selectdim(Vᵀ, 1, 1:min(η, i))
        δx = _matvec!!(δx, Tcache, _U, _Vᵀ, fx)
        @bb @. δx *= -1

        @bb copyto!(xo, x)
        @bb copyto!(fo, fx)
    end

    return build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end

# Non-allocating StaticArrays version of SimpleLimitedMemoryBroyden is actually quite
# finicky, so we'll implement it separately from the generic version
# Ignore termination_condition. Don't pass things into internal functions
function __static_solve(prob::NonlinearProblem{<:SArray}, alg::SimpleLimitedMemoryBroyden,
        args...; abstol = nothing, maxiters = 1000, kwargs...)
    x = prob.u0
    fx = _get_fx(prob, x)
    threshold = __get_threshold(alg)

    U, Vᵀ = __init_low_rank_jacobian(vec(x), vec(fx), threshold)

    abstol = __get_tolerance(x, abstol, eltype(x))

    xo, δx, fo, δf = x, -fx, fx, fx

    ls_cache = __use_linesearch(alg) === Val(true) ? LiFukushimaLineSearch()(prob, fx, x) :
               nothing

    T = promote_type(eltype(x), eltype(fx))
    if alg.alpha === nothing
        fx_norm = NONLINEARSOLVE_DEFAULT_NORM(fx)
        x_norm = NONLINEARSOLVE_DEFAULT_NORM(x)
        init_α = ifelse(fx_norm ≥ 1e-5, max(x_norm, T(true)) / (2 * fx_norm), T(true))
    else
        init_α = inv(alg.alpha)
    end

    converged, res = __unrolled_lbroyden_initial_iterations(
        prob, xo, fo, δx, abstol, U, Vᵀ, threshold, ls_cache, init_α)

    converged &&
        return build_solution(prob, alg, res.x, res.fx; retcode = ReturnCode.Success)

    xo, fo, δx = res.x, res.fx, res.δx

    for i in 1:(maxiters - _unwrap_val(threshold))
        α = ls_cache === nothing ? true : ls_cache(xo, δx)
        x = xo .+ α .* δx
        fx = prob.f(x, prob.p)
        δf = fx - fo

        maximum(abs, fx) ≤ abstol &&
            return build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)

        vᵀ = _restructure(x, _rmatvec!!(U, Vᵀ, vec(δx), init_α))
        mvec = _restructure(x, _matvec!!(U, Vᵀ, vec(δf), init_α))

        d = dot(vᵀ, δf)
        δx = @. (δx - mvec) / d

        U = Base.setindex(U, vec(δx), mod1(i, _unwrap_val(threshold)))
        Vᵀ = Base.setindex(Vᵀ, vec(vᵀ), mod1(i, _unwrap_val(threshold)))

        δx = -_restructure(fx, _matvec!!(U, Vᵀ, vec(fx), init_α))

        xo = x
        fo = fx
    end

    return build_solution(prob, alg, xo, fo; retcode = ReturnCode.MaxIters)
end

@generated function __unrolled_lbroyden_initial_iterations(
        prob, xo, fo, δx, abstol, U, Vᵀ, ::Val{threshold},
        ls_cache, init_α) where {threshold}
    calls = []
    for i in 1:threshold
        static_idx, static_idx_p1 = Val(i - 1), Val(i)
        push!(calls, quote
            α = ls_cache === nothing ? true : ls_cache(xo, δx)
            x = xo .+ α .* δx
            fx = prob.f(x, prob.p)
            δf = fx - fo

            maximum(abs, fx) ≤ abstol && return true, (; x, fx, δx)

            _U = __first_n_getindex(U, $(static_idx))
            _Vᵀ = __first_n_getindex(Vᵀ, $(static_idx))

            vᵀ = _restructure(x, _rmatvec!!(_U, _Vᵀ, vec(δx), init_α))
            mvec = _restructure(x, _matvec!!(_U, _Vᵀ, vec(δf), init_α))

            d = dot(vᵀ, δf)
            δx = @. (δx - mvec) / d

            U = Base.setindex(U, vec(δx), $(i))
            Vᵀ = Base.setindex(Vᵀ, vec(vᵀ), $(i))

            _U = __first_n_getindex(U, $(static_idx_p1))
            _Vᵀ = __first_n_getindex(Vᵀ, $(static_idx_p1))
            δx = -_restructure(fx, _matvec!!(_U, _Vᵀ, vec(fx), init_α))

            xo = x
            fo = fx
        end)
    end
    push!(calls, quote
        # Termination Check
        maximum(abs, fx) ≤ abstol && return true, (; x, fx, δx)

        return false, (; x, fx, δx)
    end)
    return Expr(:block, calls...)
end

function _rmatvec!!(y, xᵀU, U, Vᵀ, x)
    # xᵀ × (-I + UVᵀ)
    η = size(U, 2)
    if η == 0
        @bb @. y = -x
        return y
    end
    x_ = vec(x)
    xᵀU_ = view(xᵀU, 1:η)
    @bb xᵀU_ = transpose(U) × x_
    @bb y = transpose(Vᵀ) × vec(xᵀU_)
    @bb @. y -= x
    return y
end

@inline _rmatvec!!(::Nothing, Vᵀ, x, init_α) = -x .* init_α
@inline _rmatvec!!(U, Vᵀ, x, init_α) = __mapTdot(__mapdot(x, U), Vᵀ) .- x .* init_α

function _matvec!!(y, Vᵀx, U, Vᵀ, x)
    # (-I + UVᵀ) × x
    η = size(U, 2)
    if η == 0
        @bb @. y = -x
        return y
    end
    x_ = vec(x)
    Vᵀx_ = view(Vᵀx, 1:η)
    @bb Vᵀx_ = Vᵀ × x_
    @bb y = U × vec(Vᵀx_)
    @bb @. y -= x
    return y
end

@inline _matvec!!(::Nothing, Vᵀ, x, init_α) = -x .* init_α
@inline _matvec!!(U, Vᵀ, x, init_α) = __mapTdot(__mapdot(x, Vᵀ), U) .- x .* init_α

function __mapdot(x::SVector{S1}, Y::SVector{S2, <:SVector{S1}}) where {S1, S2}
    return map(Base.Fix1(dot, x), Y)
end
@generated function __mapTdot(x::SVector{S1}, Y::SVector{S1, <:SVector{S2}}) where {S1, S2}
    calls = []
    syms = [gensym("m$(i)") for i in 1:S1]
    for i in 1:S1
        push!(calls, :($(syms[i]) = x[$(i)] .* Y[$i]))
    end
    push!(calls, :(return .+($(syms...))))
    return Expr(:block, calls...)
end

@generated function __first_n_getindex(x::SVector{L, T}, ::Val{N}) where {L, T, N}
    @assert N ≤ L
    getcalls = ntuple(i -> :(x[$i]), N)
    N == 0 && return :(return nothing)
    return :(return SVector{$N, $T}(($(getcalls...))))
end

__lbroyden_threshold_cache(x, ::Val{threshold}) where {threshold} = __similar(x, threshold)
function __lbroyden_threshold_cache(x::StaticArray, ::Val{threshold}) where {threshold}
    return zeros(MArray{Tuple{threshold}, eltype(x)})
end
__lbroyden_threshold_cache(x::SArray, ::Val{threshold}) where {threshold} = nothing

function __init_low_rank_jacobian(u::StaticArray{S1, T1}, fu::StaticArray{S2, T2},
        ::Val{threshold}) where {S1, S2, T1, T2, threshold}
    T = promote_type(T1, T2)
    fuSize, uSize = Size(fu), Size(u)
    Vᵀ = MArray{Tuple{threshold, prod(uSize)}, T}(undef)
    U = MArray{Tuple{prod(fuSize), threshold}, T}(undef)
    return U, Vᵀ
end
@generated function __init_low_rank_jacobian(u::SVector{Lu, T1}, fu::SVector{Lfu, T2},
        ::Val{threshold}) where {Lu, Lfu, T1, T2, threshold}
    T = promote_type(T1, T2)
    inner_inits_Vᵀ = [:(zeros(SVector{$Lu, $T})) for i in 1:threshold]
    inner_inits_U = [:(zeros(SVector{$Lfu, $T})) for i in 1:threshold]
    return quote
        Vᵀ = SVector($(inner_inits_Vᵀ...))
        U = SVector($(inner_inits_U...))
        return U, Vᵀ
    end
end
function __init_low_rank_jacobian(u, fu, ::Val{threshold}) where {threshold}
    Vᵀ = __similar(u, threshold, length(u))
    U = __similar(u, length(fu), threshold)
    return U, Vᵀ
end
