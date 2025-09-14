"""
    $(TYPEDSIGNATURES)

Create and return the appropriate `HomotopySystemWrapper` to use for solving the given
`prob` with `alg`.
"""
function homotopy_continuation_preprocessing(
        prob::NonlinearProblem, alg::HomotopyContinuationJL)
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

    variant = iip ? Inplace : isscalar ? Scalar : OutOfPlace

    # jacobian handling
    if SciMLBase.has_jac(f.f)
        # use if present
        jac = ExplicitJacobian{variant}(f.f.f, f.f.jac)
    else
        # prepare a DI jacobian if not
        jac = construct_jacobian(f.f.f, alg.autodiff, variant, u0, p)
    end

    # variables for HC to use
    vars = if isscalar
        HC.variables(:x)
    else
        HC.variables(:x, axes(u0)...)
    end

    taylorvars = if isscalar
        [TaylorScalar(ntuple(Returns(0.0), 4)), TaylorScalar(ntuple(Returns(0.0), 4))]
    elseif iip
        (
            [TaylorScalar(ntuple(Returns(0.0), 4)) for _ in 1:(2length(u0))],
            [TaylorScalar(ntuple(Returns(0.0), 4)) for _ in 1:(2length(u0))]
        )
    else
        [TaylorScalar(ntuple(Returns(0.0), 4)) for _ in 1:(2length(u0))]
    end

    # HC-compatible system
    hcsys = HomotopySystemWrapper{variant}(
        f.f.f, jac, p, vars, taylorvars)

    return f, hcsys
end

function CommonSolve.solve(prob::NonlinearProblem, alg::HomotopyContinuationJL{true, ComplexRoots};
        denominator_abstol = 1e-7, kwargs...) where {ComplexRoots}
    f, hcsys = homotopy_continuation_preprocessing(prob, alg)

    u0 = state_values(prob)
    p = parameter_values(prob)
    isscalar = u0 isa Number

    orig_sol = HC.solve(hcsys; alg.kwargs..., kwargs...)
    only_real_roots = ComplexRoots === Val{false}
    realsols = HC.results(orig_sol; only_real = only_real_roots)
    # no solutions
    if isempty(realsols)
        retcode = SciMLBase.ReturnCode.ConvergenceFailure
        resid = NonlinearSolveBase.Utils.evaluate_f(prob, u0)
        nlsol = SciMLBase.build_solution(prob, alg, u0, resid; retcode, original = orig_sol)
        return SciMLBase.EnsembleSolution([nlsol], 0.0, false, nothing)
    end
    T = ComplexRoots === Val{false} ? eltype(u0) : promote_type(eltype(u0), Complex{real(eltype(u0))})
    validsols = isscalar ? T[] : Vector{T}[]
    for result in realsols
        # ignore ones which make the denominator zero
        test_u = ComplexRoots === Val{false} ? real.(result.solution) : result.solution
        if isscalar
            test_u = only(test_u)
        end
        if any(<=(denominator_abstol) ∘ abs, f.denominator(test_u, p))
            continue
        end
        # unpack solutions
        u = isscalar ? only(result.solution) : result.solution
        u_for_unpolynom = ComplexRoots === Val{false} ? real.(u) : u
        unpolysols = f.unpolynomialize(u_for_unpolynom, p)
        for sol in unpolysols
            any(isnan, sol) && continue
            push!(validsols, sol)
        end
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

function CommonSolve.solve(prob::NonlinearProblem, alg::HomotopyContinuationJL{false, ComplexRoots};
        denominator_abstol = 1e-7, kwargs...) where {ComplexRoots}
    f, hcsys = homotopy_continuation_preprocessing(prob, alg)

    u0 = state_values(prob)
    p = parameter_values(prob)

    u0_p = f.polynomialize(u0, p)
    fu0 = NonlinearSolveBase.Utils.evaluate_f(prob, u0_p)

    homotopy = GuessHomotopy(hcsys, fu0)
    orig_sol = HC.solve(
        homotopy, u0_p isa Number ? [[u0_p]] : [u0_p]; alg.kwargs..., kwargs...)
    only_real_roots = ComplexRoots === Val{false}
    realsols = map(res -> res.solution, HC.results(orig_sol; only_real = only_real_roots))
    if u0 isa Number
        realsols = map(only, realsols)
    end

    # no solutions or infeasible solution
    if isempty(realsols) ||
       any(<=(denominator_abstol), map(abs, f.denominator(ComplexRoots === Val{false} ? real.(only(realsols)) : only(realsols), p)))
        retcode = if isempty(realsols)
            SciMLBase.ReturnCode.ConvergenceFailure
        else
            SciMLBase.ReturnCode.Infeasible
        end
        resid = NonlinearSolveBase.Utils.evaluate_f(prob, u0)
        return SciMLBase.build_solution(prob, alg, u0, resid; retcode, original = orig_sol)
    end

    realsol = ComplexRoots === Val{false} ? real(only(realsols)) : only(realsols)
    T = eltype(u0)
    validsols = f.unpolynomialize(realsol, p)
    _, idx = findmin(validsols) do sol
        any(isnan, sol) ? Inf : norm(sol - u0_p)
    end

    u = ComplexRoots === Val{false} ? map(real, validsols[idx]) : validsols[idx]

    if any(isnan, u)
        retcode = SciMLBase.ReturnCode.Infeasible
        resid = NonlinearSolveBase.Utils.evaluate_f(prob, u0)
        return SciMLBase.build_solution(prob, alg, u0, resid; retcode, original = orig_sol)
    end

    if u0 isa Number
        u = only(u)
    end
    resid = NonlinearSolveBase.Utils.evaluate_f(prob, u)

    retcode = SciMLBase.ReturnCode.Success
    return SciMLBase.build_solution(prob, alg, u, resid; retcode, original = orig_sol)
end
