function SciMLBase.solve(
        prob::NonlinearLeastSquaresProblem{<:Union{Number, <:AbstractArray}, iip,
            <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}}},
        alg::AbstractSimpleNonlinearSolveAlgorithm,
        args...;
        kwargs...) where {T, V, P, iip}
    sol, partials = __nlsolve_ad(prob, alg, args...; kwargs...)
    dual_soln = __nlsolve_dual_soln(sol.u, partials, prob.p)
    return SciMLBase.build_solution(
        prob, alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original)
end

function SciMLBase.solve(
        prob::NonlinearProblem{<:Union{Number, <:AbstractArray}, iip,
            <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}}},
        alg::AbstractSimpleNonlinearSolveAlgorithm,
        args...;
        kwargs...) where {T, V, P, iip}
    prob = convert(ImmutableNonlinearProblem, prob)
    sol, partials = __nlsolve_ad(prob, alg, args...; kwargs...)
    dual_soln = __nlsolve_dual_soln(sol.u, partials, prob.p)
    return SciMLBase.build_solution(
        prob, alg, dual_soln, sol.resid; sol.retcode, sol.stats, sol.original)
end

for algType in (Bisection, Brent, Alefeld, Falsi, ITP, Ridder)
    @eval begin
        function SciMLBase.solve(
                prob::IntervalNonlinearProblem{
                    uType, iip, <:Union{<:Dual{T, V, P}, <:AbstractArray{<:Dual{T, V, P}}}},
                alg::$(algType),
                args...;
                kwargs...) where {uType, T, V, P, iip}
            sol, partials = __nlsolve_ad(prob, alg, args...; kwargs...)
            dual_soln = __nlsolve_dual_soln(sol.u, partials, prob.p)
            return SciMLBase.build_solution(
                prob, alg, dual_soln, sol.resid; sol.retcode, sol.stats,
                sol.original, left = Dual{T, V, P}(sol.left, partials),
                right = Dual{T, V, P}(sol.right, partials))
        end
    end
end

function __nlsolve_ad(
        prob::Union{IntervalNonlinearProblem, NonlinearProblem, ImmutableNonlinearProblem},
        alg, args...; kwargs...)
    p = value(prob.p)
    if prob isa IntervalNonlinearProblem
        tspan = value.(prob.tspan)
        newprob = IntervalNonlinearProblem(prob.f, tspan, p; prob.kwargs...)
    else
        newprob = remake(prob; p, u0 = value(prob.u0))
    end

    sol = solve(newprob, alg, args...; kwargs...)

    uu = sol.u
    f_p = __nlsolve_∂f_∂p(prob, prob.f, uu, p)
    f_x = __nlsolve_∂f_∂u(prob, prob.f, uu, p)

    z_arr = -f_x \ f_p

    pp = prob.p
    sumfun = ((z, p),) -> map(zᵢ -> zᵢ * ForwardDiff.partials(p), z)
    if uu isa Number
        partials = sum(sumfun, zip(z_arr, pp))
    elseif p isa Number
        partials = sumfun((z_arr, pp))
    else
        partials = sum(sumfun, zip(eachcol(z_arr), pp))
    end

    return sol, partials
end

function __nlsolve_ad(prob::NonlinearLeastSquaresProblem, alg, args...; kwargs...)
    newprob = remake(prob; p = value(prob.p), u0 = value(prob.u0))
    sol = solve(newprob, alg, args...; kwargs...)
    uu = sol.u

    # First check for custom `vjp` then custom `Jacobian` and if nothing is provided use
    # nested autodiff as the last resort
    if SciMLBase.has_vjp(prob.f)
        if isinplace(prob)
            _F = @closure (du, u, p) -> begin
                resid = __similar(du, length(sol.resid))
                prob.f(resid, u, p)
                prob.f.vjp(du, resid, u, p)
                du .*= 2
                return nothing
            end
        else
            _F = @closure (u, p) -> begin
                resid = prob.f(u, p)
                return reshape(2 .* prob.f.vjp(resid, u, p), size(u))
            end
        end
    elseif SciMLBase.has_jac(prob.f)
        if isinplace(prob)
            _F = @closure (du, u, p) -> begin
                J = __similar(du, length(sol.resid), length(u))
                prob.f.jac(J, u, p)
                resid = __similar(du, length(sol.resid))
                prob.f(resid, u, p)
                mul!(reshape(du, 1, :), vec(resid)', J, 2, false)
                return nothing
            end
        else
            _F = @closure (u, p) -> begin
                return reshape(2 .* vec(prob.f(u, p))' * prob.f.jac(u, p), size(u))
            end
        end
    else
        if isinplace(prob)
            _F = @closure (du, u, p) -> begin
                _f = @closure (du, u) -> prob.f(du, u, p)
                resid = __similar(du, length(sol.resid))
                v, J = DI.value_and_jacobian(_f, resid, AutoForwardDiff(), u)
                mul!(reshape(du, 1, :), vec(v)', J, 2, false)
                return nothing
            end
        else
            # For small problems, nesting ForwardDiff is actually quite fast
            if __is_extension_loaded(Val(:Zygote)) && (length(uu) + length(sol.resid) ≥ 50)
                # TODO: Remove once DI has the value_and_pullback_split defined
                _F = @closure (u, p) -> begin
                    _f = Base.Fix2(prob.f, p)
                    return __zygote_compute_nlls_vjp(_f, u, p)
                end
            else
                _F = @closure (u, p) -> begin
                    _f = Base.Fix2(prob.f, p)
                    v, J = DI.value_and_jacobian(_f, AutoForwardDiff(), u)
                    return reshape(2 .* vec(v)' * J, size(u))
                end
            end
        end
    end

    f_p = __nlsolve_∂f_∂p(prob, _F, uu, newprob.p)
    f_x = __nlsolve_∂f_∂u(prob, _F, uu, newprob.p)

    z_arr = -f_x \ f_p

    pp = prob.p
    sumfun = ((z, p),) -> map(zᵢ -> zᵢ * ForwardDiff.partials(p), z)
    if uu isa Number
        partials = sum(sumfun, zip(z_arr, pp))
    elseif pp isa Number
        partials = sumfun((z_arr, pp))
    else
        partials = sum(sumfun, zip(eachcol(z_arr), pp))
    end

    return sol, partials
end

@inline function __nlsolve_∂f_∂p(prob, f::F, u, p) where {F}
    if isinplace(prob)
        __f = p -> begin
            du = __similar(u, promote_type(eltype(u), eltype(p)))
            f(du, u, p)
            return du
        end
    else
        __f = Base.Fix1(f, u)
    end
    if p isa Number
        return __reshape(ForwardDiff.derivative(__f, p), :, 1)
    elseif u isa Number
        return __reshape(ForwardDiff.gradient(__f, p), 1, :)
    else
        return ForwardDiff.jacobian(__f, p)
    end
end

@inline function __nlsolve_∂f_∂u(prob, f::F, u, p) where {F}
    if isinplace(prob)
        __f = @closure (du, u) -> f(du, u, p)
        return ForwardDiff.jacobian(__f, __similar(u), u)
    else
        __f = Base.Fix2(f, p)
        u isa Number && return ForwardDiff.derivative(__f, u)
        return ForwardDiff.jacobian(__f, u)
    end
end

@inline function __nlsolve_dual_soln(u::Number, partials,
        ::Union{<:AbstractArray{<:Dual{T, V, P}}, Dual{T, V, P}}) where {T, V, P}
    return Dual{T, V, P}(u, partials)
end

@inline function __nlsolve_dual_soln(u::AbstractArray, partials,
        ::Union{<:AbstractArray{<:Dual{T, V, P}}, Dual{T, V, P}}) where {T, V, P}
    _partials = _restructure(u, partials)
    return map(((uᵢ, pᵢ),) -> Dual{T, V, P}(uᵢ, pᵢ), zip(u, _partials))
end
