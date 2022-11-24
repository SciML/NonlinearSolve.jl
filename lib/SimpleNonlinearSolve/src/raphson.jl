struct SimpleNewtonRaphson{CS, AD, FDT} <: AbstractNewtonAlgorithm{CS, AD, FDT}
    function SimpleNewtonRaphson(; chunk_size = Val{0}(), autodiff = Val{true}(),
                                 diff_type = Val{:forward})
        new{SciMLBase._unwrap_val(chunk_size), SciMLBase._unwrap_val(autodiff),
            SciMLBase._unwrap_val(diff_type)}()
    end
end

function SciMLBase.solve(prob::NonlinearProblem,
                         alg::SimpleNewtonRaphson, args...; abstol = nothing,
                         reltol = nothing,
                         maxiters = 1000, kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    x = float(prob.u0)
    fx = float(prob.u0)
    T = typeof(x)

    if SciMLBase.isinplace(prob)
        error("SimpleNewtonRaphson currently only supports out-of-place nonlinear problems")
    end

    atol = abstol !== nothing ? abstol : oneunit(eltype(T)) * (eps(one(eltype(T))))^(4 // 5)
    rtol = reltol !== nothing ? reltol : eps(one(eltype(T)))^(4 // 5)

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
            dfx = FiniteDiff.finite_difference_jacobian(f, x, diff_type(alg), eltype(x), fx)
        else
            fx = f(x)
            dfx = FiniteDiff.finite_difference_derivative(f, x, diff_type(alg), eltype(x),
                                                          fx)
        end
        iszero(fx) &&
            return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)
        Δx = dfx \ fx
        x -= Δx
        if isapprox(x, xo, atol = atol, rtol = rtol)
            return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)
        end
        xo = x
    end
    return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
