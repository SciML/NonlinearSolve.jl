using NonlinearSolveFirstOrder, LineSearch, ADTypes, LinearSolve
using LineSearches: LineSearches
using ForwardDiff, FiniteDiff, Zygote

include("../../../common/common_nlls_testing.jl")

linesearches = []
for lsmethod in [
        LineSearches.Static(), LineSearches.HagerZhang(), LineSearches.MoreThuente(),
        LineSearches.StrongWolfe(), LineSearches.BackTracking(),
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
append!(
    solvers,
    [
        LevenbergMarquardt(),
        LevenbergMarquardt(; linsolve = LUFactorization()),
        LevenbergMarquardt(; linsolve = KrylovJL_GMRES()),
        LevenbergMarquardt(; linsolve = KrylovJL_LSMR()),
    ]
)
# GeodesicAcceleration requests `shared > 1` from its inner descent, exercising the
# NLLS NewtonDescent init path
push!(
    solvers,
    GeneralizedFirstOrderAlgorithm(;
        descent = GeodesicAcceleration(NewtonDescent(), 0.1, 0.75),
        trustregion = NonlinearSolveFirstOrder.LevenbergMarquardtTrustRegion(1.0),
        name = :GeodesicGaussNewton, concrete_jac = Val(true)
    )
)
for radius_update_scheme in [
        RadiusUpdateSchemes.Simple, RadiusUpdateSchemes.NocedalWright,
        RadiusUpdateSchemes.NLsolve, RadiusUpdateSchemes.Hei,
        RadiusUpdateSchemes.Yuan, RadiusUpdateSchemes.Fan, RadiusUpdateSchemes.Bastin,
    ]
    push!(solvers, TrustRegion(; radius_update_scheme))
end

export solvers
