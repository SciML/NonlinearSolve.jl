using NonlinearSolveFirstOrder, SciMLBase, LinearAlgebra, Test

# The residual is only finite for u[1] >= -0.5, so a too-large trust radius steps
# into a NaN region. A NaN actual reduction used to poison ρ: every comparison in
# the radius-update branches is false for NaN, so the rejected step left the radius
# frozen and the solver stalled on the same trial point.
function nan_basin_resid(u, p)
    u[1] < -0.5 && return fill(NaN, 2)
    return [sqrt(u[1] + 0.5) - 0.5, u[2] - 4.0]
end

@testset "trust region shrinks away from a nonfinite residual region" begin
    prob = NonlinearLeastSquaresProblem(nan_basin_resid, [4.5, 4.0])
    @testset "$name" for (name, solver) in (
            ("More", TrustRegion()),
            (
                "Simple",
                TrustRegion(;
                    subproblem = TrustRegionSubproblem.Dogleg,
                    radius_update_scheme = RadiusUpdateSchemes.Simple,
                    initial_trust_radius = 10.0
                ),
            ),
            (
                "NLsolve",
                TrustRegion(;
                    subproblem = TrustRegionSubproblem.Dogleg,
                    radius_update_scheme = RadiusUpdateSchemes.NLsolve,
                    initial_trust_radius = 10.0
                ),
            ),
            (
                "NocedalWright",
                TrustRegion(;
                    subproblem = TrustRegionSubproblem.Dogleg,
                    radius_update_scheme = RadiusUpdateSchemes.NocedalWright,
                    initial_trust_radius = 10.0
                ),
            ),
            (
                "Hei",
                TrustRegion(;
                    subproblem = TrustRegionSubproblem.Dogleg,
                    radius_update_scheme = RadiusUpdateSchemes.Hei,
                    initial_trust_radius = 10.0
                ),
            ),
            (
                "Yuan",
                TrustRegion(;
                    subproblem = TrustRegionSubproblem.Dogleg,
                    radius_update_scheme = RadiusUpdateSchemes.Yuan
                ),
            ),
            (
                "Fan",
                TrustRegion(;
                    subproblem = TrustRegionSubproblem.Dogleg,
                    radius_update_scheme = RadiusUpdateSchemes.Fan
                ),
            ),
            (
                "Bastin",
                TrustRegion(;
                    subproblem = TrustRegionSubproblem.Dogleg,
                    radius_update_scheme = RadiusUpdateSchemes.Bastin
                ),
            ),
        )
        sol = solve(prob, solver; maxiters = 100)
        @test SciMLBase.successful_retcode(sol)
        @test norm(sol.resid) < 1.0e-8
    end
end
