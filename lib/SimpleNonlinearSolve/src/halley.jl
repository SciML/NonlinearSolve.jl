"""
```julia
Halley(; chunk_size = Val{0}(), autodiff = Val{true}(),
                                 diff_type = Val{:forward})
```

A low-overhead implementation of Halley's Method. This method is non-allocating on scalar
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
struct Halley{CS, AD, FDT} <: AbstractNewtonAlgorithm{CS, AD, FDT}
    function Halley(; chunk_size = Val{0}(), autodiff = Val{true}(),
                    diff_type = Val{:forward})
        new{SciMLBase._unwrap_val(chunk_size), SciMLBase._unwrap_val(autodiff),
            SciMLBase._unwrap_val(diff_type)}()
    end
end

function SciMLBase.__solve(prob::NonlinearProblem,
                           alg::Halley, args...; abstol = nothing,
                           reltol = nothing,
                           maxiters = 1000, kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    x = float(prob.u0)
    # Defining all derivative expressions in one place before the iterations
    if isa(x, AbstractArray)
        if alg_autodiff(alg)
            n = length(x)
            a_dfdx(x) = ForwardDiff.jacobian(f, x)
            a_d2fdx(x) = ForwardDiff.jacobian(a_dfdx, x)
            A = Array{Union{Nothing, Number}}(nothing, n, n)
            #fx = f(x)
        else
            n = length(x)
            f_dfdx(x) = FiniteDiff.finite_difference_jacobian(f, x, diff_type(alg), eltype(x))
            f_d2fdx(x) = FiniteDiff.finite_difference_jacobian(f_dfdx, x, diff_type(alg), eltype(x))
            A = Array{Union{Nothing, Number}}(nothing, n, n)
        end
    elseif isa(x, Number)
        if alg_autodiff(alg)
            sa_dfdx(x) = ForwardDiff.derivative(f, x)
            sa_d2fdx(x) = ForwardDiff.derivative(sa_dfdx, x)
        else
            sf_dfdx(x) = FiniteDiff.finite_difference_derivative(f, x, diff_type(alg), eltype(x))
            sf_d2fdx(x) = FiniteDiff.finite_difference_derivative(sf_dfdx, x, diff_type(alg), eltype(x))
        end
    end
    T = typeof(x)

    if SciMLBase.isinplace(prob)
        error("Halley currently only supports out-of-place nonlinear problems")
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
            if isa(x, Number)
                fx = f(x)
                dfx = sa_dfdx(x)
                d2fx = sa_d2fdx(x)
            else
                fx = f(x)
                dfx = a_dfdx(x)
                d2fx = reshape(a_d2fdx(x), (n,n,n)) # A 3-dim Hessian Tensor
                ai = -(dfx \ fx)
                for j in 1:n
                    tmp = transpose(d2fx[:, :, j] * ai)
                    A[j, :] = tmp
                end
                bi = (dfx) \ (A * ai)
                ci = (ai .* ai) ./ (ai .+ (0.5 .* bi))
            end
        else
            if isa(x, Number)
                fx = f(x)
                dfx = sf_dfdx(x)
                d2fx = sf_d2fdx(x)
            else
                fx = f(x)
                dfx = f_dfdx(x)
                d2fx = reshape(f_d2fdx(x), (n,n,n)) # A 3-dim Hessian Tensor
                ai = -(dfx \ fx)
                for j in 1:n
                    tmp = transpose(d2fx[:, :, j] * ai)
                    A[j, :] = tmp
                end
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
