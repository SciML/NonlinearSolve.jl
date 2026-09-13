function bounded_freudenstein_roth!(r, u, p)
    x, y = u
    r[1] = -13 + x + ((5 - y) * y - 2) * y
    r[2] = -29 + x + ((y + 1) * y - 14) * y
    return nothing
end

function bounded_freudenstein_roth_jac!(J, u, p)
    J[:, 1] .= 1
    J[1, 2] = -3 * u[2]^2 + 10 * u[2] - 2
    J[2, 2] = 3 * u[2]^2 + 2 * u[2] - 14
    return nothing
end

function bounded_wood!(r, u, p)
    r[1] = 10 * (u[2] - u[1]^2)
    r[2] = 1 - u[1]
    r[3] = sqrt(90) * (u[4] - u[3]^2)
    r[4] = 1 - u[3]
    r[5] = sqrt(10) * (u[2] + u[4] - 2)
    r[6] = (u[2] - u[4]) / sqrt(10)
    return nothing
end

function bounded_wood_jac!(J, u, p)
    fill!(J, 0)
    J[1, 1], J[1, 2], J[2, 1] = -20 * u[1], 10, -1
    J[3, 3], J[3, 4], J[4, 3] = -2 * sqrt(90) * u[3], sqrt(90), -1
    J[5, 2], J[5, 4] = sqrt(10), sqrt(10)
    J[6, 2], J[6, 4] = 1 / sqrt(10), -1 / sqrt(10)
    return nothing
end

function bounded_bard!(r, u, p)
    for i in eachindex(p)
        r[i] = p[i] - u[1] - i / (u[2] * (16 - i) + u[3] * min(i, 16 - i))
    end
    return nothing
end

function bounded_bard_jac!(J, u, p)
    for i in eachindex(p)
        v, w = 16 - i, min(i, 16 - i)
        d = u[2] * v + u[3] * w
        J[i, 1], J[i, 2], J[i, 3] = -1, i * v / d^2, i * w / d^2
    end
    return nothing
end

function bounded_kowalik!(r, u, p)
    @. r = p.y - u[1] * (p.x^2 + u[2] * p.x) / (p.x^2 + u[3] * p.x + u[4])
    return nothing
end

function bounded_kowalik_jac!(J, u, p)
    for (i, x) in enumerate(p.x)
        a, b = x^2 + u[2] * x, x^2 + u[3] * x + u[4]
        J[i, 1], J[i, 2] = -a / b, -u[1] * x / b
        J[i, 3], J[i, 4] = u[1] * a * x / b^2, u[1] * a / b^2
    end
    return nothing
end

function bounded_validation_cases()
    cases = BoundedBenchmarkCase[]
    for (i, u0) in enumerate(([0.5, -2.0], [10.0, -1.0], [1.0, 1.0], [4.0, 3.0]))
        push!(
            cases, bounded_case(
                "freudenstein_roth_$(i)", "nonconvex_root",
                bounded_freudenstein_roth!, bounded_freudenstein_roth_jac!, u0, nothing,
                [-5.0, -5.0], [20.0, 5.0]; root = true
            )
        )
    end
    for (i, u0) in enumerate(([-3.0, -1.0, -3.0, -1.0], [-1.2, 1.0, -1.2, 1.0], [2.0, 3.0, 2.0, 3.0], [0.0, 0.5, 0.0, 0.5]))
        push!(
            cases, bounded_case(
                "wood_$(i)", "wood", bounded_wood!, bounded_wood_jac!,
                u0, nothing, [-3.0, -1.0, -3.0, -1.0], [3.0, 4.0, 3.0, 4.0]; prototype = zeros(6, 4)
            )
        )
    end
    y_bard = [0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58, 0.73, 0.96, 1.34, 2.1, 4.39]
    for (i, u0) in enumerate(([1.0, 1.0, 1.0], [0.1, 5.0, 5.0], [0.0, 0.1, 0.1], [4.0, 9.0, 0.1]))
        push!(
            cases, bounded_case(
                "bard_$(i)", "nonzero_residual_fit", bounded_bard!,
                bounded_bard_jac!, u0, y_bard, [0.0, 0.01, 0.01], [5.0, 10.0, 10.0];
                prototype = zeros(15, 3), reference_cost = 0.00410743865328949
            )
        )
    end
    p = (;
        x = [4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625],
        y = [0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246],
    )
    for (i, u0) in enumerate(([0.25, 0.39, 0.415, 0.39], [1.0, 1.0, 1.0, 1.0], [0.01, 4.0, 4.0, 0.01], [0.5, 0.01, 0.01, 4.0]))
        push!(
            cases, bounded_case(
                "kowalik_$(i)", "nonzero_residual_fit", bounded_kowalik!,
                bounded_kowalik_jac!, u0, p, fill(0.001, 4), fill(5.0, 4);
                prototype = zeros(11, 4), reference_cost = 0.00015375280192461965
            )
        )
    end
    ad_cases = (
        "rosenbrock_2_false_1", "rosenbrock_20_false_1", "rosenbrock_100_false_1",
        "rosenbrock_2_true_2", "exponential_3.0_1", "exponential_0.2_3",
    )
    for case in bounded_benchmark_cases()
        case.name in ad_cases || continue
        prob = case.prob
        push!(
            cases, bounded_case(
                "ad_$(case.name)", "automatic_jacobian", prob.f.f,
                case.jacobian, prob.u0, prob.p, prob.lb, prob.ub; root = prob isa NonlinearProblem,
                prototype = prob.f.jac_prototype, reference_cost = case.reference_cost, analytic = false
            )
        )
    end
    return cases
end
