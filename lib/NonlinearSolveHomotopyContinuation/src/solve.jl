function homotopy_continuation_preprocessing(prob::NonlinearProblem, alg::HomotopyContinuationJL)
    # cast to a `HomotopyNonlinearFunction`
    f = if prob.f isa HomotopyNonlinearFunction
        prob.f
    else
        HomotopyNonlinearFunction(prob.f)
    end

    u0 = state_values(prob)
    p = parameter_values(prob)
    isscalar = u0 isa Number
    iip = SciMLBase.isinplace(prob)

    # jacobian handling
    if SciMLBase.has_jac(f)
        # use if present
        prep = nothing
    elseif iip
        # prepare a DI jacobian if not
        prep = DI.prepare_jacobian(prob.f.f, copy(state_values(prob)), alg.autodiff, u0, DI.Constant(p))
    elseif isscalar
        prep = DI.prepare_derivative(prob.f.f, alg.autodiff, u0, DI.Constant(p))
    else
        prep = DI.prepare_jacobian(prob.f.f, alg.autodiff, u0, DI.Constant(p))
    end

    # variables for HC to use
    vars = if isscalar
        HC.variables(:x)
    else
        HC.variables(:x, axes(u0)...)
    end

    # TODO: Is there an upper bound for the order?
    taylorvars = if isscalar
        Taylor1(zeros(ComplexF64, 5), 4)
    elseif iip
        ([Taylor1(zeros(ComplexF64, 5), 4) for _ in u0], [Taylor1(zeros(ComplexF64, 5), 4) for _ in u0])
    else
        [Taylor1(zeros(ComplexF64, 5), 4) for _ in u0]
    end

    jacobian_buffers = if isscalar
        nothing
    elseif iip
        (similar(u0), similar(u0), similar(u0, length(u0), length(u0)))
    else
        (similar(u0), similar(u0, length(u0), length(u0)))
    end

    # HC-compatible system
    variant = iip ? Inplace : isscalar ? Scalar : OutOfPlace
    hcsys = HomotopySystemWrapper{variant}(prob, alg.autodiff, prep, vars, taylorvars, jacobian_buffers)

    return f, hcsys
end

function CommonSolve.solve(prob::NonlinearProblem, alg::HomotopyContinuationJL{true}; denominator_abstol = 1e-7, kwargs...)
    f, hcsys = homotopy_continuation_preprocessing(prob, alg)

    u0 = state_values(prob)
    p = parameter_values(prob)
    isscalar = u0 isa Number

    orig_sol = HC.solve(hcsys; alg.kwargs..., kwargs...)
    realsols = HC.results(orig_sol; only_real = true)
    # no real solutions
    if isempty(realsols)
        retcode = SciMLBase.ReturnCode.ConvergenceFailure
        resid = NonlinearSolveBase.Utils.evaluate_f(prob, u0)
        nlsol = SciMLBase.build_solution(prob, alg, u0, resid; retcode, original = orig_sol)
        return SciMLBase.EnsembleSolution([nlsol], 0.0, false, nothing)
    end
    T = eltype(u0)
    validsols = isscalar ? T[] : Vector{T}[]
    for result in realsols
        # ignore ones which make the denominator zero
        real_u = real.(result.solution)
        if isscalar
            real_u = only(real_u)
        end
        if any(<=(denominator_abstol) ∘ abs, f.denominator(real_u, p))
            continue
        end
        # unpack solutions and make them real
        u = isscalar ? only(result.solution) : result.solution
        append!(validsols, f.unpolynomialize(real.(u), p))
    end

    # if there are no valid solutions
    if isempty(validsols)
        retcode = SciMLBase.ReturnCode.Infeasible
        resid = NonlinearSolveBase.Utils.evaluate_f(prob, u0)
        nlsol = SciMLBase.build_solution(prob, alg, u0, resid; retcode, original = orig_sol)
        return SciMLBase.EnsembleSolution([nlsol], 0.0, false, nothing)
    end

    retcode = SciMLBase.ReturnCode.Success
    nlsols = map(validsols) do u
        resid = NonlinearSolveBase.Utils.evaluate_f(prob, u)
        return SciMLBase.build_solution(prob, alg, u, resid; retcode, original = orig_sol)
    end
    return SciMLBase.EnsembleSolution(nlsols, 0.0, true, nothing)
end

function CommonSolve.solve(prob::NonlinearProblem, alg::HomotopyContinuationJL{false}; denominator_abstol = 1e-7, kwargs...)
    f, hcsys = homotopy_continuation_preprocessing(prob, alg)

    u0 = state_values(prob)
    p = parameter_values(prob)

    u0_p = f.polynomialize(u0, p)
    fu0 = NonlinearSolveBase.Utils.evaluate_f(prob, u0_p)

    homotopy = GuessHomotopy(hcsys, fu0)
    if u0_p isa Number
        u0_p = [u0_p]
    end
    orig_sol = HC.solve(homotopy, [u0_p]; alg.kwargs..., kwargs...)
    realsols = HC.results(orig_sol; only_real = true)

    # no real solutions or infeasible solution
    if isempty(realsols) || any(<=(denominator_abstol), f.denominator(real.(only(realsols).solution), p))
        retcode = if isempty(realsols)
            SciMLBase.ReturnCode.ConvergenceFailure
        else
            SciMLBase.ReturnCode.Infeasible
        end
        resid = NonlinearSolveBase.Utils.evaluate_f(prob, u0)
        return SciMLBase.build_solution(prob, alg, u0, resid; retcode, original = orig_sol)
    end
    realsol = only(realsols)
    T = eltype(u0)
    validsols = f.unpolynomialize(realsol.solution, p)
    sol, idx = findmin(validsols) do sol
        norm(sol - u0)
    end
    resid = NonlinearSolveBase.Utils.evaluate_f(prob, u0)

    retcode = SciMLBase.ReturnCode.Success
    return SciMLBase.build_solution(prob, alg, u0, resid; retcode, original = orig_sol)
end

