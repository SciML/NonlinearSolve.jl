using NonlinearSolveFirstOrder
include("setup_corerootfindtesting.jl")

damping_initial = [0.5, 2.0, 5.0]
damping_increase_factor = [1.5, 3.0, 10.0]
damping_decrease_factor = Float64[2, 5, 10.0]
finite_diff_step_geodesic = [0.02, 0.2, 0.3]
α_geodesic = [0.6, 0.8, 0.9]
b_uphill = Float64[0, 1, 2]
min_damping_D = [1.0e-12, 1.0e-9, 1.0e-4]

list_of_options = zip(
    damping_initial, damping_increase_factor, damping_decrease_factor,
    finite_diff_step_geodesic, α_geodesic, b_uphill, min_damping_D
)
for options in list_of_options
    alg = LevenbergMarquardt(;
        damping_initial = options[1], damping_increase_factor = options[2],
        damping_decrease_factor = options[3],
        finite_diff_step_geodesic = options[4], α_geodesic = options[5],
        b_uphill = options[6], min_damping_D = options[7]
    )

    sol = solve_oop(quadratic_f, [1.0, 1.0], 2.0; solver = alg, maxiters = 10000)
    @test SciMLBase.successful_retcode(sol)
    err = maximum(abs, quadratic_f(sol.u, 2.0))
    @test err < 1.0e-9
end
