module NonlinearSolveEnlsipExt

using FastClosures: @closure
using NonlinearSolve: NonlinearSolve, EnlsipJL
using SciMLBase: SciMLBase, NonlinearLeastSquaresProblem, ReturnCode
using Enlsip: Enlsip

function SciMLBase.__solve(prob::NonlinearLeastSquaresProblem, alg::EnlsipJL, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000,
        alias_u0::Bool = false, maxtime = nothing, show_trace::Val{ST} = Val(false),
        termination_condition = nothing, kwargs...) where {ST}
    NonlinearSolve.__test_termination_condition(termination_condition, :EnlsipJL)

    f, u0, resid = NonlinearSolve.__construct_extension_f(
        prob; alias_u0, can_handle_oop = Val(true), force_oop = Val(true))

    f_aug = @closure u -> begin
        u_ = view(u, 1:(length(u) - 1))
        r = f(u_)
        return vcat(r, u[length(u)])
    end

    eq_cons = u -> [u[end]]

    abstol = NonlinearSolve.DEFAULT_TOLERANCE(abstol, eltype(u0))
    reltol = NonlinearSolve.DEFAULT_TOLERANCE(reltol, eltype(u0))

    maxtime = maxtime === nothing ? typemax(eltype(u0)) : maxtime

    jac_fn = NonlinearSolve.__construct_extension_jac(
        prob, alg, u0, resid; alg.autodiff, can_handle_oop = Val(true))

    n = length(u0) + 1
    m = length(resid) + 1

    jac_fn_aug = @closure u -> begin
        u_ = view(u, 1:(length(u) - 1))
        J = jac_fn(u_)
        J_full = similar(u, (m, n))
        J_full[1:(m - 1), 1:(n - 1)] .= J
        fill!(J_full[1:(m - 1), n], false)
        fill!(J_full[m, 1:(n - 1)], false)
        return J_full
    end

    u0 = vcat(u0, 0.0)
    u_low = [eltype(u0)(ifelse(i == 1, -Inf, 0)) for i in 1:length(u0)]
    u_up = [eltype(u0)(ifelse(i == 1, Inf, 0)) for i in 1:length(u0)]

    model = Enlsip.CnlsModel(
        f_aug, n, m; starting_point = u0, jacobian_residuals = jac_fn_aug,
        x_low = u_low, x_upp = u_up, nb_eqcons = 1, eq_constraints = eq_cons)
    Enlsip.solve!(model; max_iter = maxiters, time_limit = maxtime, silent = !ST,
        abs_tol = abstol, rel_tol = reltol, x_tol = reltol)

    sol_u = Enlsip.solution(model)
    resid = Enlsip.sum_sq_residuals(model)

    status = Enlsip.status(model)
    retcode = status === :found_first_order_stationary_point ? ReturnCode.Success :
              status === :maximum_iterations_exceeded ? ReturnCode.MaxIters :
              status === :time_limit_exceeded ? ReturnCode.MaxTime : ReturnCode.Failure

    return SciMLBase.build_solution(prob, alg, sol_u, resid; retcode, original = model)
end

end
