using NonlinearSolveFirstOrder
include("setup_corerootfindtesting.jl")

max_trust_radius = [10.0, 100.0, 1000.0]
initial_trust_radius = [10.0, 1.0, 0.1]
step_threshold = [0.0, 0.01, 0.25]
shrink_threshold = [0.25, 0.3, 0.5]
expand_threshold = [0.5, 0.8, 0.9]
shrink_factor = [0.1, 0.3, 0.5]
expand_factor = [1.5, 2.0, 3.0]
max_shrink_times = [10, 20, 30]

list_of_options = zip(
    max_trust_radius, initial_trust_radius, step_threshold, shrink_threshold,
    expand_threshold, shrink_factor, expand_factor, max_shrink_times
)

for options in list_of_options
    alg = TrustRegion(;
        max_trust_radius = options[1], initial_trust_radius = options[2],
        step_threshold = options[3], shrink_threshold = options[4],
        expand_threshold = options[5], shrink_factor = options[6],
        expand_factor = options[7], max_shrink_times = options[8]
    )

    sol = solve_oop(quadratic_f, [1.0, 1.0], 2.0; solver = alg)
    @test SciMLBase.successful_retcode(sol)
    err = maximum(abs, quadratic_f(sol.u, 2.0))
    @test err < 1.0e-9
end
