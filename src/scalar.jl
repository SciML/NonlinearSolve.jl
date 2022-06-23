function SciMLBase.solve(prob::NonlinearProblem{<:Union{Number, SVector}},
                         alg::NewtonRaphson, args...; xatol = nothing, xrtol = nothing,
                         maxiters = 1000, kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    x = float(prob.u0)
    fx = float(prob.u0)
    T = typeof(x)
    atol = xatol !== nothing ? xatol : oneunit(eltype(T)) * (eps(one(eltype(T))))^(4 // 5)
    rtol = xrtol !== nothing ? xrtol : eps(one(eltype(T)))^(4 // 5)

    if typeof(x) <: Number
        xo = oftype(one(eltype(x)), Inf)
    else
        xo = map(x -> oftype(one(eltype(x)), Inf), x)
    end

    for i in 1:maxiters
        if alg_autodiff(alg)
            fx, dfx = value_derivative(f, x)
        elseif x isa AbstractArray
            fx = f(x)
            dfx = FiniteDiff.finite_difference_jacobian(f, x, alg.diff_type, eltype(x), fx)
        else
            fx = f(x)
            dfx = FiniteDiff.finite_difference_derivative(f, x, alg.diff_type, eltype(x),
                                                          fx)
        end
        iszero(fx) &&
            return SciMLBase.build_solution(prob, alg, x, fx; retcode = Symbol(DEFAULT))
        Δx = dfx \ fx
        x -= Δx
        if isapprox(x, xo, atol = atol, rtol = rtol)
            return SciMLBase.build_solution(prob, alg, x, fx; retcode = Symbol(DEFAULT))
        end
        xo = x
    end
    return SciMLBase.build_solution(prob, alg, x, fx; retcode = Symbol(MAXITERS_EXCEED))
end

function scalar_nlsolve_ad(prob, alg, args...; kwargs...)
    f = prob.f
    p = value(prob.p)
    u0 = value(prob.u0)

    newprob = NonlinearProblem(f, u0, p; prob.kwargs...)
    sol = solve(newprob, alg, args...; kwargs...)

    uu = sol.u
    if p isa Number
        f_p = ForwardDiff.derivative(Base.Fix1(f, uu), p)
    else
        f_p = ForwardDiff.gradient(Base.Fix1(f, uu), p)
    end

    f_x = ForwardDiff.derivative(Base.Fix2(f, p), uu)
    pp = prob.p
    sumfun = let f_x′ = -f_x
        ((fp, p),) -> (fp / f_x′) * ForwardDiff.partials(p)
    end
    partials = sum(sumfun, zip(f_p, pp))
    return sol, partials
end

function SciMLBase.solve(prob::NonlinearProblem{<:Union{Number, SVector}, iip,
                                                <:Dual{T, V, P}}, alg::NewtonRaphson,
                         args...; kwargs...) where {iip, T, V, P}
    sol, partials = scalar_nlsolve_ad(prob, alg, args...; kwargs...)
    return SciMLBase.build_solution(prob, alg, Dual{T, V, P}(sol.u, partials), sol.resid;
                                    retcode = sol.retcode)
end
function SciMLBase.solve(prob::NonlinearProblem{<:Union{Number, SVector}, iip,
                                                <:AbstractArray{<:Dual{T, V, P}}},
                         alg::NewtonRaphson, args...; kwargs...) where {iip, T, V, P}
    sol, partials = scalar_nlsolve_ad(prob, alg, args...; kwargs...)
    return SciMLBase.build_solution(prob, alg, Dual{T, V, P}(sol.u, partials), sol.resid;
                                    retcode = sol.retcode)
end

# avoid ambiguities
for Alg in [Bisection]
    @eval function SciMLBase.solve(prob::NonlinearProblem{uType, iip, <:Dual{T, V, P}},
                                   alg::$Alg, args...;
                                   kwargs...) where {uType, iip, T, V, P}
        sol, partials = scalar_nlsolve_ad(prob, alg, args...; kwargs...)
        return SciMLBase.build_solution(prob, alg, Dual{T, V, P}(sol.u, partials),
                                        sol.resid; retcode = sol.retcode,
                                        left = Dual{T, V, P}(sol.left, partials),
                                        right = Dual{T, V, P}(sol.right, partials))
        #return BracketingSolution(Dual{T,V,P}(sol.left, partials), Dual{T,V,P}(sol.right, partials), sol.retcode, sol.resid)
    end
    @eval function SciMLBase.solve(prob::NonlinearProblem{uType, iip,
                                                          <:AbstractArray{<:Dual{T, V, P}}},
                                   alg::$Alg, args...;
                                   kwargs...) where {uType, iip, T, V, P}
        sol, partials = scalar_nlsolve_ad(prob, alg, args...; kwargs...)
        return SciMLBase.build_solution(prob, alg, Dual{T, V, P}(sol.u, partials),
                                        sol.resid; retcode = sol.retcode,
                                        left = Dual{T, V, P}(sol.left, partials),
                                        right = Dual{T, V, P}(sol.right, partials))
        #return BracketingSolution(Dual{T,V,P}(sol.left, partials), Dual{T,V,P}(sol.right, partials), sol.retcode, sol.resid)
    end
end

function SciMLBase.solve(prob::NonlinearProblem, alg::Bisection, args...; maxiters = 1000,
                         kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    left, right = prob.u0
    fl, fr = f(left), f(right)

    if iszero(fl)
        return SciMLBase.build_solution(prob, alg, left, fl;
                                        retcode = Symbol(EXACT_SOLUTION_LEFT), left = left,
                                        right = right)
    end

    i = 1
    if !iszero(fr)
        while i < maxiters
            mid = (left + right) / 2
            (mid == left || mid == right) &&
                return SciMLBase.build_solution(prob, alg, left, fl;
                                                retcode = Symbol(FLOATING_POINT_LIMIT),
                                                left = left, right = right)
            fm = f(mid)
            if iszero(fm)
                right = mid
                break
            end
            if sign(fl) == sign(fm)
                fl = fm
                left = mid
            else
                fr = fm
                right = mid
            end
            i += 1
        end
    end

    while i < maxiters
        mid = (left + right) / 2
        (mid == left || mid == right) &&
            return SciMLBase.build_solution(prob, alg, left, fl;
                                            retcode = Symbol(FLOATING_POINT_LIMIT),
                                            left = left, right = right)
        fm = f(mid)
        if iszero(fm)
            right = mid
            fr = fm
        else
            left = mid
            fl = fm
        end
        i += 1
    end

    return SciMLBase.build_solution(prob, alg, left, fl; retcode = Symbol(MAXITERS_EXCEED),
                                    left = left, right = right)
end

function SciMLBase.solve(prob::NonlinearProblem, alg::Falsi, args...; maxiters = 1000,
                         kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    left, right = prob.u0
    fl, fr = f(left), f(right)

    if iszero(fl)
        return SciMLBase.build_solution(prob, alg, left, fl;
                                        retcode = Symbol(EXACT_SOLUTION_LEFT), left = left,
                                        right = right)
    end

    i = 1
    if !iszero(fr)
        while i < maxiters
            if nextfloat_tdir(left, prob.u0...) == right
                return SciMLBase.build_solution(prob, alg, left, fl;
                                                retcode = Symbol(FLOATING_POINT_LIMIT),
                                                left = left, right = right)
            end
            mid = (fr * left - fl * right) / (fr - fl)
            for i in 1:10
                mid = max_tdir(left, prevfloat_tdir(mid, prob.u0...), prob.u0...)
            end
            if mid == right || mid == left
                break
            end
            fm = f(mid)
            if iszero(fm)
                right = mid
                break
            end
            if sign(fl) == sign(fm)
                fl = fm
                left = mid
            else
                fr = fm
                right = mid
            end
            i += 1
        end
    end

    while i < maxiters
        mid = (left + right) / 2
        (mid == left || mid == right) &&
            return SciMLBase.build_solution(prob, alg, left, fl;
                                            retcode = Symbol(FLOATING_POINT_LIMIT),
                                            left = left, right = right)
        fm = f(mid)
        if iszero(fm)
            right = mid
            fr = fm
        elseif sign(fm) == sign(fl)
            left = mid
            fl = fm
        else
            right = mid
            fr = fm
        end
        i += 1
    end

    return SciMLBase.build_solution(prob, alg, left, fl; retcode = Symbol(MAXITERS_EXCEED),
                                    left = left, right = right)
end
