using NonlinearSolve, SciMLBase, LinearAlgebra, SparseArrays

struct BoundedBenchmarkCase{P, J}
    name::String
    family::String
    representation::Symbol
    prob::P
    reference_cost::Float64
    jacobian::J
end

function bounded_case(
        name, family, f!, jac!, u0, p, lb, ub;
        root = false, representation = :dense, prototype = nothing,
        jvp = nothing, vjp = nothing, reference_cost = 0.0, analytic = true
    )
    resid_prototype = prototype === nothing ? nothing : zeros(size(prototype, 1))
    nf = NonlinearFunction{true}(
        f!; jac = analytic ? jac! : nothing,
        jac_prototype = prototype, resid_prototype, jvp, vjp
    )
    constructor = root ? NonlinearProblem : NonlinearLeastSquaresProblem
    prob = constructor(nf, copy(u0), p; lb, ub)
    return BoundedBenchmarkCase(name, family, representation, prob, reference_cost, jac!)
end

function bounded_rosenbrock!(r, u, p)
    for i in 1:2:length(u)
        r[i] = 10 * (u[i + 1] - u[i]^2)
        r[i + 1] = 1 - u[i]
    end
    return nothing
end

function bounded_rosenbrock_jac!(J, u, p)
    fill!(J, 0)
    for i in 1:2:length(u)
        J[i, i] = -20 * u[i]
        J[i, i + 1] = 10
        J[i + 1, i] = -1
    end
    return nothing
end

function bounded_powell!(r, u, p)
    r[1] = u[1] + 10 * u[2]
    r[2] = sqrt(5) * (u[3] - u[4])
    r[3] = (u[2] - 2 * u[3])^2
    r[4] = sqrt(10) * (u[1] - u[4])^2
    return nothing
end

function bounded_powell_jac!(J, u, p)
    fill!(J, 0)
    J[1, 1], J[1, 2] = 1, 10
    J[2, 3], J[2, 4] = sqrt(5), -sqrt(5)
    J[3, 2], J[3, 3] = 2 * (u[2] - 2 * u[3]), -4 * (u[2] - 2 * u[3])
    J[4, 1], J[4, 4] = 2 * sqrt(10) * (u[1] - u[4]), -2 * sqrt(10) * (u[1] - u[4])
    return nothing
end

function bounded_beale!(r, u, p)
    for (i, a) in enumerate((1.5, 2.25, 2.625))
        r[i] = a - u[1] * (1 - u[2]^i)
    end
    return nothing
end

function bounded_beale_jac!(J, u, p)
    for i in 1:3
        J[i, 1] = u[2]^i - 1
        J[i, 2] = i * u[1] * u[2]^(i - 1)
    end
    return nothing
end

function bounded_brown!(r, u, p)
    r[1] = u[1] - 1.0e6
    r[2] = u[2] - 2.0e-6
    r[3] = u[1] * u[2] - 2
    return nothing
end

function bounded_brown_jac!(J, u, p)
    fill!(J, 0)
    J[1, 1], J[2, 2], J[3, 1], J[3, 2] = 1, 1, u[2], u[1]
    return nothing
end

function bounded_exponential!(r, u, p)
    @. r = u[1] * exp(u[2] * p.t) + u[3] - p.y
    return nothing
end

function bounded_exponential_jac!(J, u, p)
    for i in eachindex(p.t)
        e = exp(u[2] * p.t[i])
        J[i, 1], J[i, 2], J[i, 3] = e, u[1] * p.t[i] * e, 1
    end
    return nothing
end

function bounded_linear!(r, u, p)
    mul!(r, p.A, u)
    r .-= p.b
    return nothing
end

bounded_linear_jac!(J, u, p) = copyto!(J, p.A)

function bounded_diffusion!(r, u, p)
    mul!(r, p.A, u)
    @. r += p.gamma * u^3 - p.b
    return nothing
end

function bounded_diffusion_jac!(J, u, p)
    copyto!(J, p.A)
    for i in eachindex(u)
        J[i, i] += 3 * p.gamma * u[i]^2
    end
    return nothing
end

function bounded_diffusion_jvp!(out, v, u, p)
    mul!(out, p.A, v)
    @. out += 3 * p.gamma * u^2 * v
    return nothing
end

function bounded_benchmark_cases()
    cases = BoundedBenchmarkCase[]
    for n in (2, 20, 100), active in (false, true), start in (1, 2)
        lb, ub = fill(-2.0, n), fill(3.0, n)
        active && (ub[1:2:end] .= 0.5)
        u0 = repeat(start == 1 ? [-1.2, 1.0] : [0.0, 0.0], n ÷ 2)
        push!(
            cases, bounded_case(
                "rosenbrock_$(n)_$(active)_$(start)", "rosenbrock",
                bounded_rosenbrock!, bounded_rosenbrock_jac!, u0, nothing, lb, ub;
                root = !active, reference_cost = active ? n / 16 : 0.0
            )
        )
    end
    for (i, u0) in enumerate(([0.9, -0.8, 0.5, 0.8], [0.1, 0.1, 0.1, 0.1], [-1.0, 1.0, -1.0, 1.0]))
        push!(
            cases, bounded_case(
                "powell_$(i)", "singular", bounded_powell!,
                bounded_powell_jac!, u0, nothing, fill(-1.0, 4), ones(4); root = true
            )
        )
    end
    for (i, u0) in enumerate(([1.0, 0.1], [4.0, 1.0], [0.1, 0.9]))
        push!(
            cases, bounded_case(
                "beale_$(i)", "small_dense", bounded_beale!,
                bounded_beale_jac!, u0, nothing, zeros(2), [4.0, 1.0]; prototype = zeros(3, 2)
            )
        )
    end
    for (i, u0) in enumerate(([1.0, 1.0], [1.0e5, 0.1], [1.5e6, 1.0e-5]))
        push!(
            cases, bounded_case(
                "brown_$(i)", "badly_scaled", bounded_brown!,
                bounded_brown_jac!, u0, nothing, zeros(2), [2.0e6, 2.0]; prototype = zeros(3, 2)
            )
        )
    end
    for span in (0.2, 3.0), (i, u0) in enumerate(([1.0, -1.0, 0.5], [5.0, 0.0, 0.0], [0.1, -1.9, 0.9]))
        t = collect(range(0, span; length = 40))
        p = (; t, y = @. 2 * exp(-0.5 * t) + 0.1)
        push!(
            cases, bounded_case(
                "exponential_$(span)_$(i)", "curve_fit",
                bounded_exponential!, bounded_exponential_jac!, u0, p,
                [0.0, -2.0, 0.0], [5.0, 0.0, 1.0]; prototype = zeros(40, 3)
            )
        )
    end
    for (m, n, rank) in ((12, 8, 4), (4, 8, 4)), start in (0.1, 0.9)
        L = [sin(i * j) for i in 1:m, j in 1:rank]
        R = [cos(i * j) for i in 1:rank, j in 1:n]
        A = L * R
        p = (; A, b = A * collect(range(0.2, 0.8; length = n)))
        push!(
            cases, bounded_case(
                "linear_$(m)_$(n)_$(start)", "rank_deficient",
                bounded_linear!, bounded_linear_jac!, fill(start, n), p, zeros(n), ones(n);
                prototype = zeros(m, n)
            )
        )
    end
    for start in (0.1, 0.9)
        n = 20
        A = Matrix{Float64}(I, n, n)
        lb, ub = zeros(n), ones(n)
        lb[1:3:end] .= 0.4
        ub[1:3:end] .= 0.4
        target = clamp.(collect(range(-0.5, 1.5; length = n)), lb, ub)
        p = (; A, b = collect(range(-0.5, 1.5; length = n)))
        push!(
            cases, bounded_case(
                "fixed_$(start)", "fixed_variables",
                bounded_linear!, bounded_linear_jac!, clamp.(fill(start, n), lb, ub), p, lb, ub;
                reference_cost = sum(abs2, target - p.b) / 2
            )
        )
    end
    for n in (64, 256, 1024), active in (false, true), representation in (:sparse, :operator)
        A = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(3.0, n), 1 => fill(-1.0, n - 1))
        gamma = 0.1
        target = [0.5 + 0.25 * sin(i / n * pi) for i in 1:n]
        q = zeros(n)
        if active
            target[1:5:end] .= 1
            q[1:5:end] .= 0.1
            target[3:7:end] .= 0
            q[3:7:end] .= -0.1
        end
        J = A + spdiagm(0 => @. 3 * gamma * target^2)
        residual = -(J' \ q)
        p = (; A, gamma, b = A * target + gamma * target .^ 3 - residual)
        for start in (0.1, 0.9)
            push!(
                cases, bounded_case(
                    "diffusion_$(n)_$(active)_$(representation)_$(start)",
                    "diffusion", bounded_diffusion!, bounded_diffusion_jac!, fill(start, n), p,
                    zeros(n), ones(n); root = !active, representation, prototype = copy(A),
                    jvp = bounded_diffusion_jvp!, vjp = bounded_diffusion_jvp!,
                    reference_cost = sum(abs2, residual) / 2
                )
            )
        end
    end
    return cases
end

function bounded_benchmark_metrics(case, sol)
    prob = case.prob
    u = sol.u
    r = similar(sol.resid)
    prob.f(r, u, prob.p)
    J = prob.f.jac_prototype === nothing ? zeros(length(r), length(u)) : copy(prob.f.jac_prototype)
    case.jacobian(J, u, prob.p)
    gradient = J' * r
    projected = u .- clamp.(u .- gradient, prob.lb, prob.ub)
    cost = sum(abs2, r) / 2
    feasible = all(prob.lb .<= u .<= prob.ub)
    root = prob isa NonlinearProblem
    accurate = root ? maximum(abs, r) <= 1.0e-6 :
        maximum(abs, projected) <= 1.0e-5 && cost <= case.reference_cost + 1.0e-8 * max(1, case.reference_cost)
    return (;
        passed = SciMLBase.successful_retcode(sol) && feasible && accurate,
        feasible, cost, residual = maximum(abs, r), stationarity = maximum(abs, projected),
    )
end
