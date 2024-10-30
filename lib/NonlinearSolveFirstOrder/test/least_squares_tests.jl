@testsetup module CoreNLLSTesting

using NonlinearSolveFirstOrder, LineSearch, ADTypes, LinearSolve
using LineSearches: LineSearches
using ForwardDiff, FiniteDiff, Zygote

include("../../../common/common_nlls_testing.jl")

linesearches = []
for lsmethod in [
    LineSearches.Static(), LineSearches.HagerZhang(), LineSearches.MoreThuente(),
    LineSearches.StrongWolfe(), LineSearches.BackTracking()
]
    push!(linesearches, LineSearchesJL(; method = lsmethod))
end
push!(linesearches, BackTracking())

solvers = []
for linsolve in [nothing, LUFactorization(), KrylovJL_GMRES(), KrylovJL_LSMR()]
    vjp_autodiffs = linsolve isa KrylovJL ? [nothing, AutoZygote(), AutoFiniteDiff()] :
                    [nothing]
    for linesearch in linesearches, vjp_autodiff in vjp_autodiffs
        push!(solvers, GaussNewton(; linsolve, linesearch, vjp_autodiff))
    end
end
append!(solvers,
    [
        LevenbergMarquardt(),
        LevenbergMarquardt(; linsolve = LUFactorization()),
        LevenbergMarquardt(; linsolve = KrylovJL_GMRES()),
        LevenbergMarquardt(; linsolve = KrylovJL_LSMR())
    ]
)
for radius_update_scheme in [
    RadiusUpdateSchemes.Simple, RadiusUpdateSchemes.NocedalWright,
    RadiusUpdateSchemes.NLsolve, RadiusUpdateSchemes.Hei,
    RadiusUpdateSchemes.Yuan, RadiusUpdateSchemes.Fan, RadiusUpdateSchemes.Bastin
]
    push!(solvers, TrustRegion(; radius_update_scheme))
end

export solvers

end

@testitem "General NLLS Solvers" setup=[CoreNLLSTesting] tags=[:core] begin
    using LinearAlgebra

    nlls_problems = [prob_oop, prob_iip, prob_oop_vjp, prob_iip_vjp]

    for prob in nlls_problems, solver in solvers
        sol = solve(prob, solver; maxiters = 10000, abstol = 1e-6)
        @test SciMLBase.successful_retcode(sol)
        @test norm(sol.resid, 2) < 1e-6
    end
end
