@testitem "Jacobian Cache Inference" tags = [:core] begin
    using ADTypes, NonlinearSolveBase, NonlinearSolveFirstOrder, SciMLBase, Test

    f_iip = NonlinearFunction{true}((du, u, p) -> du .= u .^ 2 .- p)

    u0 = [1.0, 2.0]
    p = [1.0, 4.0]
    fu = similar(u0)
    f_iip(fu, u0, p)

    prob_iip = NonlinearProblem(f_iip, u0, p)

    stats = NonlinearSolveBase.NLStats(0, 0, 0, 0, 0)
    alg = TrustRegion(autodiff = AutoFiniteDiff())
    autodiff = AutoFiniteDiff()

    @testset "In-place, no analytic Jacobian" begin
        @inferred NonlinearSolveBase.construct_jacobian_cache(
            prob_iip, alg, prob_iip.f, fu, u0, p; stats, autodiff
        )
    end

    @testset "With analytic Jacobian" begin
        f_jac = NonlinearFunction{true}(
            (du, u, p) -> du .= u .^ 2 .- p;
            jac = (J, u, p) -> (
                J[1, 1] = 2u[1]; J[2, 2] = 2u[2];
                J[1, 2] = 0; J[2, 1] = 0
            ),
        )
        prob_jac = NonlinearProblem(f_jac, u0, p)
        @inferred NonlinearSolveBase.construct_jacobian_cache(
            prob_jac, alg, prob_jac.f, fu, u0, p; stats, autodiff
        )
    end

    @testset "Full init: TrustRegion + AutoFiniteDiff" begin
        @inferred SciMLBase.init(prob_iip, TrustRegion(autodiff = AutoFiniteDiff()))
    end

    @testset "Full init: NewtonRaphson + AutoFiniteDiff" begin
        @inferred SciMLBase.init(prob_iip, NewtonRaphson(autodiff = AutoFiniteDiff()))
    end
end
