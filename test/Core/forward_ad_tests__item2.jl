using NonlinearSolve

using SciMLBase
using ForwardDiff, FiniteDiff

function objfn(F, init, params)
    th1, th2 = init
    px, py, l1, l2 = params
    F[1] = l1 * cos(th1) + l2 * cos(th1 + th2) - px
    F[2] = l1 * sin(th1) + l2 * sin(th1 + th2) - py
    return F
end

function solve_nlprob(pxpy)
    px, py = pxpy
    theta1 = pi / 4
    theta2 = pi / 4
    initial_guess = [theta1; theta2]
    l1 = 60
    l2 = 60
    p = [px; py; l1; l2]
    prob = NonlinearLeastSquaresProblem(
        NonlinearFunction(objfn, resid_prototype = zeros(2)),
        initial_guess, p
    )
    resu = solve(
        prob,
        reltol = 1.0e-12, abstol = 1.0e-12
    )
    th1, th2 = resu.u
    cable1_base = [-90; 0; 0]
    cable2_base = [-150; 0; 0]
    cable3_base = [150; 0; 0]
    cable1_top = [l1 * cos(th1) / 2; l1 * sin(th1) / 2; 0]
    cable23_top = [
        l1 * cos(th1) + l2 * cos(th1 + th2) / 2;
        l1 * sin(th1) + l2 * sin(th1 + th2) / 2; 0
    ]
    c1_length = sqrt(
        (cable1_top[1] - cable1_base[1])^2 +
            (cable1_top[2] - cable1_base[2])^2
    )
    c2_length = sqrt(
        (cable23_top[1] - cable2_base[1])^2 +
            (cable23_top[2] - cable2_base[2])^2
    )
    c3_length = sqrt(
        (cable23_top[1] - cable3_base[1])^2 +
            (cable23_top[2] - cable3_base[2])^2
    )
    return c1_length + c2_length + c3_length
end

grad1 = ForwardDiff.gradient(solve_nlprob, [34.0, 87.0])
grad2 = FiniteDiff.finite_difference_gradient(solve_nlprob, [34.0, 87.0])

@test grad1 ≈ grad2 atol = 1.0e-3

hess1 = ForwardDiff.hessian(solve_nlprob, [34.0, 87.0])
hess2 = FiniteDiff.finite_difference_hessian(solve_nlprob, [34.0, 87.0])

@test hess1 ≈ hess2 atol = 1.0e-3

function solve_nlprob_with_cache(pxpy)
    px, py = pxpy
    theta1 = pi / 4
    theta2 = pi / 4
    initial_guess = [theta1; theta2]
    l1 = 60
    l2 = 60
    p = [px; py; l1; l2]
    prob = NonlinearLeastSquaresProblem(
        NonlinearFunction(objfn, resid_prototype = zeros(2)),
        initial_guess, p
    )
    cache = init(prob; reltol = 1.0e-12, abstol = 1.0e-12)
    resu = solve!(cache)
    th1, th2 = resu.u
    cable1_base = [-90; 0; 0]
    cable2_base = [-150; 0; 0]
    cable3_base = [150; 0; 0]
    cable1_top = [l1 * cos(th1) / 2; l1 * sin(th1) / 2; 0]
    cable23_top = [
        l1 * cos(th1) + l2 * cos(th1 + th2) / 2;
        l1 * sin(th1) + l2 * sin(th1 + th2) / 2; 0
    ]
    c1_length = sqrt(
        (cable1_top[1] - cable1_base[1])^2 +
            (cable1_top[2] - cable1_base[2])^2
    )
    c2_length = sqrt(
        (cable23_top[1] - cable2_base[1])^2 +
            (cable23_top[2] - cable2_base[2])^2
    )
    c3_length = sqrt(
        (cable23_top[1] - cable3_base[1])^2 +
            (cable23_top[2] - cable3_base[2])^2
    )
    return c1_length + c2_length + c3_length
end

grad1 = ForwardDiff.gradient(solve_nlprob_with_cache, [34.0, 87.0])
grad2 = FiniteDiff.finite_difference_gradient(solve_nlprob_with_cache, [34.0, 87.0])

@test grad1 ≈ grad2 atol = 1.0e-3

hess1 = ForwardDiff.hessian(solve_nlprob_with_cache, [34.0, 87.0])
hess2 = FiniteDiff.finite_difference_hessian(solve_nlprob_with_cache, [34.0, 87.0])

@test hess1 ≈ hess2 atol = 1.0e-3

function singular_nlls_resid!(resid, u, p)
    resid[1] = u[1] - p[1]
    resid[2] = 0
    return nothing
end

function singular_nlls_solution(p)
    prob = NonlinearLeastSquaresProblem(
        NonlinearFunction(singular_nlls_resid!, resid_prototype = zeros(2)),
        [1.0, 2.0], p
    )
    return solve(prob, NewtonRaphson(); abstol = 1.0e-12, reltol = 1.0e-12)
end

singular_nlls_sum(p) = sum(singular_nlls_solution(p).u)

function singular_nlls_cached_solution(p)
    prob = NonlinearLeastSquaresProblem(
        NonlinearFunction(singular_nlls_resid!, resid_prototype = zeros(2)),
        [1.0, 2.0], p
    )
    cache = init(prob, NewtonRaphson(); abstol = 1.0e-12, reltol = 1.0e-12)
    return solve!(cache)
end

singular_nlls_cached_sum(p) = sum(singular_nlls_cached_solution(p).u)

@test SciMLBase.successful_retcode(singular_nlls_solution([1.0]).retcode)
@test SciMLBase.successful_retcode(singular_nlls_cached_solution([1.0]).retcode)
@test ForwardDiff.gradient(singular_nlls_sum, [1.0]) ≈ [1.0]
@test ForwardDiff.gradient(singular_nlls_cached_sum, [1.0]) ≈ [1.0]

function scalar_nl_solution(p)
    prob = NonlinearProblem((u, p) -> u^2 - p[1] - p[2], 2.0, p)
    return solve(prob, NewtonRaphson(); abstol = 1.0e-12, reltol = 1.0e-12)
end

scalar_nl_sum(p) = scalar_nl_solution(p).u

@test SciMLBase.successful_retcode(scalar_nl_solution([4.0, 0.0]).retcode)
@test ForwardDiff.gradient(scalar_nl_sum, [4.0, 0.0]) ≈ [0.25, 0.25]
