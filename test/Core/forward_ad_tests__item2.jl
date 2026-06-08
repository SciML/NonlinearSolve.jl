using NonlinearSolve

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
