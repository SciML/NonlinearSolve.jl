"""
```julia
SimpleHalley(; chunk_size = Val{0}(), autodiff = Val{true}(),
                                 diff_type = Val{:forward})
```

A low-overhead implementation of SimpleHalley's Method. This method is non-allocating on scalar
and static array problems.

!!! note

    As part of the decreased overhead, this method omits some of the higher level error
    catching of the other methods. Thus, to see better error messages, use one of the other
    methods like `NewtonRaphson`

### Keyword Arguments

- `chunk_size`: the chunk size used by the internal ForwardDiff.jl automatic differentiation
  system. This allows for multiple derivative columns to be computed simultaneously,
  improving performance. Defaults to `0`, which is equivalent to using ForwardDiff.jl's
  default chunk size mechanism. For more details, see the documentation for
  [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/).
- `autodiff`: whether to use forward-mode automatic differentiation for the Jacobian.
  Note that this argument is ignored if an analytical Jacobian is passed; as that will be
  used instead. Defaults to `Val{true}`, which means ForwardDiff.jl is used by default.
  If `Val{false}`, then FiniteDiff.jl is used for finite differencing.
- `diff_type`: the type of finite differencing used if `autodiff = false`. Defaults to
  `Val{:forward}` for forward finite differences. For more details on the choices, see the
  [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl) documentation.
"""
struct SimpleHalley{CS, AD, FDT} <: AbstractNewtonAlgorithm{CS, AD, FDT}
    function SimpleHalley(; chunk_size = Val{0}(), autodiff = Val{true}(),
        diff_type = Val{:forward})
        new{SciMLBase._unwrap_val(chunk_size), SciMLBase._unwrap_val(autodiff),
            SciMLBase._unwrap_val(diff_type)}()
    end
end

function SciMLBase.__solve(prob::NonlinearProblem,
    alg::SimpleHalley, args...; abstol = nothing,
    reltol = nothing,
    maxiters = 1000, kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    x = float(prob.u0)
    fx = f(x)
    if isa(x, AbstractArray)
        n = length(x)
    end
    T = typeof(x)

    if SciMLBase.isinplace(prob)
        error("SimpleHalley currently only supports out-of-place nonlinear problems")
    end

    atol = abstol !== nothing ? abstol :
           real(oneunit(eltype(T))) * (eps(real(one(eltype(T)))))^(4 // 5)
    rtol = reltol !== nothing ? reltol : eps(real(one(eltype(T))))^(4 // 5)

    if x isa Number
        xo = oftype(one(eltype(x)), Inf)
    else
        xo = map(x -> oftype(one(eltype(x)), Inf), x)
    end

    for i in 1:maxiters
        if alg_autodiff(alg)
            if isa(x, Number)
                fx = f(x)
                dfx = ForwardDiff.derivative(f, x)
                d2fx = ForwardDiff.derivative(x -> ForwardDiff.derivative(f, x), x)
            else
                fx = f(x)
                dfx = ForwardDiff.jacobian(f, x)
                d2fx = ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x)
                ai = -(dfx \ fx)
                A = reshape(d2fx * ai, (n, n))
                bi = (dfx) \ (A * ai)
                ci = (ai .* ai) ./ (ai .+ (0.5 .* bi))
            end
        else
            if isa(x, Number)
                fx = f(x)
                dfx = FiniteDiff.finite_difference_derivative(f, x, diff_type(alg),
                    eltype(x))
                d2fx = FiniteDiff.finite_difference_derivative(x -> FiniteDiff.finite_difference_derivative(f,
                        x),
                    x,
                    diff_type(alg), eltype(x))
            else
                fx = f(x)
                dfx = FiniteDiff.finite_difference_jacobian(f, x, diff_type(alg), eltype(x))
                d2fx = FiniteDiff.finite_difference_jacobian(x -> FiniteDiff.finite_difference_jacobian(f,
                        x),
                    x,
                    diff_type(alg), eltype(x))
                ai = -(dfx \ fx)
                A = reshape(d2fx * ai, (n, n))
                bi = (dfx) \ (A * ai)
                ci = (ai .* ai) ./ (ai .+ (0.5 .* bi))
            end
        end
        iszero(fx) &&
            return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)
        if isa(x, Number)
            Δx = (2 * dfx^2 - fx * d2fx) \ (2fx * dfx)
            x -= Δx
        else
            Δx = ci
            x += Δx
        end
        if isapprox(x, xo, atol = atol, rtol = rtol)
            return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)
        end
        xo = x
    end

    return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
