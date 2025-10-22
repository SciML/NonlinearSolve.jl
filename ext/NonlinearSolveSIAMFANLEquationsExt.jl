module NonlinearSolveSIAMFANLEquationsExt

using FastClosures: @closure
using SIAMFANLEquations: SIAMFANLEquations, aasol, nsol, nsoli, nsolsc, ptcsol, ptcsoli,
                         ptcsolsc, secant

using NonlinearSolveBase: NonlinearSolveBase
using NonlinearSolve: NonlinearSolve, SIAMFANLEquationsJL
using SciMLBase: SciMLBase, NonlinearProblem, ReturnCode

function siamfanlequations_retcode_mapping(sol)
    if sol.errcode == 0
        return ReturnCode.Success
    elseif sol.errcode == 10
        return ReturnCode.MaxIters
    elseif sol.errcode == 1
        return ReturnCode.Failure
    elseif sol.errcode == -1
        return ReturnCode.Default
    else
        error(lazy"Unknown SIAMFANLEquations return code: $(sol.errcode)")
    end
end

function zeros_like(x, args...)
    z = similar(x, args...)
    fill!(z, false)
    return z
end

# pseudo transient continuation has a fixed cost per iteration, iteration statistics are
# not interesting here.
function siamfanlequations_stats_mapping(method, sol)
    ((method === :pseudotransient) || (method === :anderson)) && return nothing
    return SciMLBase.NLStats(
        sum(sol.stats.ifun), sum(sol.stats.ijac), 0, 0, sum(sol.stats.iarm)
    )
end

function SciMLBase.__solve(
        prob::NonlinearProblem, alg::SIAMFANLEquationsJL, args...;
        abstol = nothing, reltol = nothing, alias_u0::Bool = false, maxiters = 1000,
        termination_condition = nothing, show_trace = Val(false), kwargs...
)
    NonlinearSolveBase.assert_extension_supported_termination_condition(
        termination_condition, alg
    )

    (; method, delta, linsolve, m, beta) = alg
    T = eltype(prob.u0)
    atol = NonlinearSolveBase.get_tolerance(abstol, T)
    rtol = NonlinearSolveBase.get_tolerance(reltol, T)

    printerr = show_trace isa Val{true}

    if prob.u0 isa Number
        f = Base.Fix2(prob.f, prob.p)
        if method == :newton
            sol = nsolsc(f, prob.u0; maxit = maxiters, atol, rtol, printerr)
        elseif method == :pseudotransient
            sol = ptcsolsc(
                f, prob.u0; delta0 = delta, maxit = maxiters, atol, rtol, printerr
            )
        elseif method == :secant
            sol = secant(f, prob.u0; maxit = maxiters, atol, rtol, printerr)
        elseif method == :anderson
            f_aa, u,
            _ = NonlinearSolveBase.construct_extension_function_wrapper(
                prob; alias_u0, make_fixed_point = Val(true)
            )
            sol = aasol(
                f_aa, u, m, zeros_like(u, 1, 2 * m + 4);
                maxit = maxiters, atol, rtol, beta
            )
        end
    else
        f, u,
        resid = NonlinearSolveBase.construct_extension_function_wrapper(
            prob; alias_u0, make_fixed_point = Val(method == :anderson)
        )
        N = length(u)
        FS = zeros_like(u, N)

        # Jacobian Free Newton Krylov
        if linsolve !== nothing
            # Allocate ahead for Krylov basis
            JVS = linsolve == :gmres ? zeros_like(u, N, 3) : zeros_like(u, N)
            linsolve_alg = String(linsolve)
            if method == :newton
                sol = nsoli(
                    f, u, FS, JVS; lsolver = linsolve_alg,
                    maxit = maxiters, atol, rtol, printerr
                )
            elseif method == :pseudotransient
                sol = ptcsoli(
                    f, u, FS, JVS; lsolver = linsolve_alg,
                    maxit = maxiters, atol, rtol, printerr
                )
            end
        else
            if prob.f.jac === nothing && alg.autodiff === missing
                FPS = zeros_like(u, N, N)
                if method == :newton
                    sol = nsol(
                        f, u, FS, FPS; sham = 1, atol, rtol, maxit = maxiters, printerr
                    )
                elseif method == :pseudotransient
                    sol = ptcsol(
                        f, u, FS, FPS;
                        atol, rtol, maxit = maxiters, delta0 = delta, printerr
                    )
                elseif method == :anderson
                    sol = aasol(
                        f, u, m, zeros(T, N, 2 * m + 4);
                        atol, rtol, maxit = maxiters, beta
                    )
                end
            else
                autodiff = alg.autodiff === missing ? nothing : alg.autodiff
                FPS = prob.f.jac_prototype !== nothing ? zero(prob.f.jac_prototype) :
                      zeros_like(u, N, N)
                jac = NonlinearSolveBase.construct_extension_jac(
                    prob, alg, u, resid; autodiff
                )
                AJ! = @closure (J, u, x) -> jac(J, x)
                if method == :newton
                    sol = nsol(
                        f, u, FS, FPS, AJ!; sham = 1, atol,
                        rtol, maxit = maxiters, printerr
                    )
                elseif method == :pseudotransient
                    sol = ptcsol(
                        f, u, FS, FPS, AJ!; atol, rtol, maxit = maxiters,
                        delta0 = delta, printerr
                    )
                end
            end
        end
    end

    retcode = siamfanlequations_retcode_mapping(sol)
    stats = siamfanlequations_stats_mapping(method, sol)
    res = prob.u0 isa Number && method === :anderson ? sol.solution[1] : sol.solution
    resid = NonlinearSolveBase.Utils.evaluate_f(prob, res)
    return SciMLBase.build_solution(prob, alg, res, resid; retcode, stats, original = sol)
end

end
