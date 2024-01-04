module NonlinearSolveSIAMFANLEquationsExt

using NonlinearSolve, SIAMFANLEquations, SciMLBase
import FastClosures: @closure

@inline function __siam_fanl_equations_retcode_mapping(sol)
    if sol.errcode == 0
        return ReturnCode.Success
    elseif sol.errcode == 10
        return ReturnCode.MaxIters
    elseif sol.errcode == 1
        return ReturnCode.Failure
    elseif sol.errcode == -1
        return ReturnCode.Default
    end
end

@inline function __zeros_like(x, args...)
    z = similar(x, args...)
    fill!(z, zero(eltype(x)))
    return z
end

# pseudo transient continuation has a fixed cost per iteration, iteration statistics are
# not interesting here.
@inline function __siam_fanl_equations_stats_mapping(method, sol)
    ((method === :pseudotransient) || (method === :anderson)) && return nothing
    return SciMLBase.NLStats(sum(sol.stats.ifun), sum(sol.stats.ijac), 0, 0,
        sum(sol.stats.iarm))
end

function SciMLBase.__solve(prob::NonlinearProblem, alg::SIAMFANLEquationsJL, args...;
        abstol = nothing, reltol = nothing, alias_u0::Bool = false, maxiters = 1000,
        termination_condition = nothing, show_trace::Val{ShT} = Val(false),
        kwargs...) where {ShT}
    NonlinearSolve.__test_termination_condition(termination_condition, :SIAMFANLEquationsJL)

    (; method, delta, linsolve, m, beta) = alg
    T = eltype(prob.u0)
    atol = NonlinearSolve.DEFAULT_TOLERANCE(abstol, T)
    rtol = NonlinearSolve.DEFAULT_TOLERANCE(reltol, T)

    if prob.u0 isa Number
        f = @closure u -> prob.f(u, prob.p)
        if method == :newton
            sol = nsolsc(f, prob.u0; maxit = maxiters, atol, rtol, printerr = ShT)
        elseif method == :pseudotransient
            sol = ptcsolsc(f, prob.u0; delta0 = delta, maxit = maxiters, atol, rtol,
                printerr = ShT)
        elseif method == :secant
            sol = secant(f, prob.u0; maxit = maxiters, atol, rtol, printerr = ShT)
        elseif method == :anderson
            f_aa, u, _ = NonlinearSolve.__construct_extension_f(prob; alias_u0,
                make_fixed_point = Val(true))
            sol = aasol(f_aa, u, m, __zeros_like(u, 1, 2 * m + 4); maxit = maxiters,
                atol, rtol, beta)
        end
    else
        f, u, resid = NonlinearSolve.__construct_extension_f(prob; alias_u0,
            make_fixed_point = Val(method == :anderson))
        N = length(u)
        FS = __zeros_like(u, N)

        # Jacobian Free Newton Krylov
        if linsolve !== nothing
            # Allocate ahead for Krylov basis
            JVS = linsolve == :gmres ? __zeros_like(u, N, 3) : __zeros_like(u, N)
            linsolve_alg = String(linsolve)
            if method == :newton
                sol = nsoli(f, u, FS, JVS; lsolver = linsolve_alg, maxit = maxiters, atol,
                    rtol, printerr = ShT)
            elseif method == :pseudotransient
                sol = ptcsoli(f, u, FS, JVS; lsolver = linsolve_alg, maxit = maxiters,
                    atol, rtol, printerr = ShT)
            end
        else
            if prob.f.jac === nothing && alg.autodiff === missing
                FPS = __zeros_like(u, N, N)
                if method == :newton
                    sol = nsol(f, u, FS, FPS; sham = 1, atol, rtol, maxit = maxiters,
                        printerr = ShT)
                elseif method == :pseudotransient
                    sol = ptcsol(f, u, FS, FPS; atol, rtol, maxit = maxiters,
                        delta0 = delta, printerr = ShT)
                elseif method == :anderson
                    sol = aasol(f, u, m, zeros(T, N, 2 * m + 4); atol, rtol,
                        maxit = maxiters, beta)
                end
            else
                FPS = prob.f.jac_prototype !== nothing ? zero(prob.f.jac_prototype) :
                      __zeros_like(u, N, N)
                jac = NonlinearSolve.__construct_extension_jac(prob, alg, u, resid;
                    alg.autodiff)
                AJ! = @closure (J, u, x) -> jac(J, x)
                if method == :newton
                    sol = nsol(f, u, FS, FPS, AJ!; sham = 1, atol, rtol, maxit = maxiters,
                        printerr = ShT)
                elseif method == :pseudotransient
                    sol = ptcsol(f, u, FS, FPS, AJ!; atol, rtol, maxit = maxiters,
                        delta0 = delta, printerr = ShT)
                end
            end
        end
    end

    retcode = __siam_fanl_equations_retcode_mapping(sol)
    stats = __siam_fanl_equations_stats_mapping(method, sol)
    res = prob.u0 isa Number && method === :anderson ? sol.solution[1] : sol.solution
    resid = NonlinearSolve.evaluate_f(prob, res)
    return SciMLBase.build_solution(prob, alg, res, resid; retcode, stats, original = sol)
end

end
