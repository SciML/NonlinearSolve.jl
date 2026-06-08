using NonlinearSolveFirstOrder
include("setup_corerootfindtesting.jl")

maxiterations = [2, 3, 4, 5]
u0 = [1.0, 1.0]
@testset for radius_update_scheme in [
        RadiusUpdateSchemes.Simple, RadiusUpdateSchemes.NocedalWright,
        RadiusUpdateSchemes.NLsolve, RadiusUpdateSchemes.Hei,
        RadiusUpdateSchemes.Yuan, RadiusUpdateSchemes.Fan, RadiusUpdateSchemes.Bastin,
    ]
    @testset for maxiters in maxiterations
        solver = TrustRegion(; radius_update_scheme)
        sol_iip = solve_iip(quadratic_f!, u0; solver, maxiters)
        sol_oop = solve_oop(quadratic_f, u0; solver, maxiters)
        @test sol_iip.u ≈ sol_oop.u
    end
end
