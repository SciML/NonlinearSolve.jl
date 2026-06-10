using NonlinearSolveSpectralMethods
include("setup_corerootfindtesting.jl")

σ_min = [1.0e-10, 1.0e-5, 1.0e-4]
σ_max = [1.0e10, 1.0e5, 1.0e4]
σ_1 = [1.0, 0.5, 2.0]
M = [10, 1, 100]
γ = [1.0e-4, 1.0e-3, 1.0e-5]
τ_min = [0.1, 0.2, 0.3]
τ_max = [0.5, 0.8, 0.9]
nexp = [2, 1, 2]
η_strategy = [
    (f_1, k, x, F) -> f_1 / k^2, (f_1, k, x, F) -> f_1 / k^3,
    (f_1, k, x, F) -> f_1 / k^4,
]

list_of_options = zip(σ_min, σ_max, σ_1, M, γ, τ_min, τ_max, nexp, η_strategy)
for options in list_of_options
    local probN, sol, alg
    alg = DFSane(;
        sigma_min = options[1], sigma_max = options[2], sigma_1 = options[3],
        M = options[4], gamma = options[5], tau_min = options[6],
        tau_max = options[7], n_exp = options[8], eta_strategy = options[9]
    )

    probN = NonlinearProblem{false}(quadratic_f, [1.0, 1.0], 2.0)
    sol = solve(probN, alg, abstol = 1.0e-11)
    @test all(abs.(quadratic_f(sol.u, 2.0)) .< 1.0e-6)
end
