"""
```julia
SimpleNewtonRaphson(; chunk_size = Val{0}(), autodiff = Val{true}(),
                                 diff_type = Val{:forward})
```

A low-overhead implementation of Newton-Raphson. This method is non-allocating on scalar
and static array problems.

!!! note

    As part of the decreased overhead, this method omits some of the higher level error
    catching of the other methods. Thus to see better error messages, use one of the other
    methods like `NewtonRaphson`

### Keyword Arguments

- `chunk_size`: the chunk size used by the internal ForwardDiff.jl automatic differentiation
  system. This allows for multiple derivative columns to be computed simultaniously,
  improving performance. Defaults to `0`, which is equivalent to using ForwardDiff.jl's
  default chunk size mechanism. For more details, see the documentation for
  [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/).
- `autodiff`: whether to use forward-mode automatic differentiation for the Jacobian.
  Note that this argument is ignored if an analytical Jacobian is passed as that will be
  used instead. Defaults to `Val{true}`, which means ForwardDiff.jl is used by default.
  If `Val{false}`, then FiniteDiff.jl is used for finite differencing.
- `diff_type`: the type of finite differencing used if `autodiff = false`. Defaults to
  `Val{:forward}` for forward finite differences. For more details on the choices, see the
  [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl) documentation.
"""
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

    atol = abstol !== nothing ? abstol :
           real(oneunit(eltype(T))) * (eps(real(one(eltype(T)))))^(4 // 5)
    rtol = reltol !== nothing ? reltol : eps(real(one(eltype(T))))^(4 // 5)

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

    @show x, fx
    return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
