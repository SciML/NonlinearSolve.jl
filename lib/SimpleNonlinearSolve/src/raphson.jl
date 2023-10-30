"""
    SimpleNewtonRaphson(; batched = false,
        chunk_size = Val{0}(),
        autodiff = Val{true}(),
        diff_type = Val{:forward},
        termination_condition = missing)

A low-overhead implementation of Newton-Raphson. This method is non-allocating on scalar
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
- `termination_condition`: control the termination of the algorithm. (Only works for batched
  problems)
"""
struct SimpleNewtonRaphson{CS, AD, FDT} <: AbstractNewtonAlgorithm{CS, AD, FDT} end

function SimpleNewtonRaphson(; batched = false,
    chunk_size = Val{0}(),
    autodiff = Val{true}(),
    diff_type = Val{:forward},
    termination_condition = missing)
    if !ismissing(termination_condition) && !batched
        throw(ArgumentError("`termination_condition` is currently only supported for batched problems"))
    end
    if batched
        # @assert ADLinearSolveFDExtLoaded[] "Please install and load `LinearSolve.jl`, `FiniteDifferences.jl` and `AbstractDifferentiation.jl` to use batched Newton-Raphson."
        termination_condition = ismissing(termination_condition) ?
                                NLSolveTerminationCondition(NLSolveTerminationMode.NLSolveDefault;
            abstol = nothing,
            reltol = nothing) :
                                termination_condition
        return BatchedSimpleNewtonRaphson(; chunk_size,
            autodiff,
            diff_type,
            termination_condition)
        return SimpleNewtonRaphson{SciMLBase._unwrap_val(chunk_size),
            SciMLBase._unwrap_val(autodiff),
            SciMLBase._unwrap_val(diff_type)}()
    end
    return SimpleNewtonRaphson{SciMLBase._unwrap_val(chunk_size),
        SciMLBase._unwrap_val(autodiff),
        SciMLBase._unwrap_val(diff_type)}()
end

const SimpleGaussNewton = SimpleNewtonRaphson

function SciMLBase.__solve(prob::Union{NonlinearProblem,NonlinearLeastSquaresProblem},
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

    if prob isa NonlinearLeastSquaresProblem && !(typeof(prob.u0) <: Union{Number, AbstractVector})
        error("SimpleGaussNewton only supports Number and AbstactVector types. Please convert any problem of AbstractArray into one with u0 as AbstractVector")
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
        if DiffEqBase.has_jac(prob.f)
            dfx = prob.f.jac(x, prob.p)
            fx = f(x)
        elseif alg_autodiff(alg)
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

        if prob isa NonlinearProblem
            Δx = _restructure(fx, dfx \ _vec(fx))
        else
            Δx = dfx \ fx
        end

        x -= Δx
        if isapprox(x, xo, atol = atol, rtol = rtol)
            return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.Success)
        end
        xo = x
    end

    return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end
