@testitem "Scalar Ops" begin
    using ADTypes, SciMLBase
    using Enzyme, Zygote, ForwardDiff, FiniteDiff, ReverseDiff, Tracker
    using SciMLJacobianOperators

    reverse_ADs = [
        AutoEnzyme(),
        AutoEnzyme(; mode = Enzyme.Reverse),
        AutoZygote(),
        # AutoReverseDiff(), # FIXME: https://github.com/gdalle/DifferentiationInterface.jl/issues/503
        AutoTracker(),
        AutoFiniteDiff()
    ]

    forward_ADs = [
        AutoEnzyme(),
        AutoEnzyme(; mode = Enzyme.Forward),
        AutoForwardDiff(),
        AutoFiniteDiff()
    ]

    prob = NonlinearProblem(NonlinearFunction{false}((u, p) -> u^2 - p), 1.0, 2.0)

    analytic_jac(u, p) = 2 * u
    analytic_jvp(v, u, p) = 2 * u * v
    analytic_vjp(v, u, p) = analytic_jvp(v, u, p)

    @testset "AutoDiff" begin
        @testset for jvp_autodiff in forward_ADs, vjp_autodiff in reverse_ADs
            jac_op = JacobianOperator(prob, -1.0, 1.0; jvp_autodiff, vjp_autodiff)

            @testset for u in rand(4), v in rand(4)
                sop = StatefulJacobianOperator(jac_op, u, prob.p)
                @test (sop * v)≈analytic_jvp(v, u, prob.p) atol=1e-5
                @test (sop' * v)≈analytic_vjp(v, u, prob.p) atol=1e-5

                normal_form_sop = sop' * sop
                JᵀJv = normal_form_sop * v
                J_analytic = analytic_jac(u, prob.p)
                JᵀJv_analytic = J_analytic' * J_analytic * v
                @test JᵀJv≈JᵀJv_analytic atol=1e-5
            end
        end
    end

    prob = NonlinearProblem(
        NonlinearFunction{false}((u, p) -> u^2 - p; jvp = analytic_jvp, vjp = analytic_vjp),
        1.0, 2.0)

    @testset "Analytic JVP/VJP" begin
        jac_op = JacobianOperator(prob, -1.0, 1.0)

        @testset for u in rand(4), v in rand(4)
            sop = StatefulJacobianOperator(jac_op, u, prob.p)
            @test (sop * v)≈analytic_jvp(v, u, prob.p) atol=1e-5
            @test (sop' * v)≈analytic_vjp(v, u, prob.p) atol=1e-5

            normal_form_sop = sop' * sop
            JᵀJv = normal_form_sop * v
            J_analytic = analytic_jac(u, prob.p)
            JᵀJv_analytic = J_analytic' * J_analytic * v
            @test JᵀJv≈JᵀJv_analytic atol=1e-5
        end
    end

    prob = NonlinearProblem(
        NonlinearFunction{false}((u, p) -> u^2 - p; jac = (u, p) -> 2 * u), 1.0, 2.0)

    @testset "Analytic Jacobian" begin
        jac_op = JacobianOperator(prob, -1.0, 1.0)

        @testset for u in rand(4), v in rand(4)
            sop = StatefulJacobianOperator(jac_op, u, prob.p)
            @test (sop * v)≈2 * u * v atol=1e-5
            @test (sop' * v)≈2 * u * v atol=1e-5

            normal_form_sop = sop' * sop
            JᵀJv = normal_form_sop * v
            J_analytic = analytic_jac(u, prob.p)
            JᵀJv_analytic = J_analytic' * J_analytic * v
            @test JᵀJv≈JᵀJv_analytic atol=1e-5
        end
    end
end

@testitem "Inplace Problems" begin
    using ADTypes, SciMLBase
    using Enzyme, ForwardDiff, FiniteDiff, ReverseDiff
    using SciMLJacobianOperators

    reverse_ADs = [
        AutoEnzyme(),
        AutoEnzyme(; mode = Enzyme.Reverse),
        # AutoReverseDiff(), # FIXME: https://github.com/gdalle/DifferentiationInterface.jl/issues/503
        AutoFiniteDiff()
    ]

    forward_ADs = [
        AutoEnzyme(),
        AutoEnzyme(; mode = Enzyme.Forward),
        AutoForwardDiff(),
        AutoFiniteDiff()
    ]

    prob = NonlinearProblem(
        NonlinearFunction{true}((du, u, p) -> du .= u .^ 2 .- p .+ u[2] * u[1]), [1.0, 3.0], 2.0)

    analytic_jac(u, p) = [2 * u[1]+u[2] u[1]; u[2] 2 * u[2]+u[1]]
    analytic_jvp(v, u, p) = analytic_jac(u, p) * v
    analytic_vjp(v, u, p) = analytic_jac(u, p)' * v

    analytic_jac!(J, u, p) = (J .= analytic_jac(u, p))
    analytic_jvp!(J, v, u, p) = (J .= analytic_jvp(v, u, p))
    analytic_vjp!(J, v, u, p) = (J .= analytic_vjp(v, u, p))

    @testset "AutoDiff" begin
        @testset for jvp_autodiff in forward_ADs, vjp_autodiff in reverse_ADs
            jac_op = JacobianOperator(prob, [2.0, 3.0], prob.u0; jvp_autodiff, vjp_autodiff)

            @testset for u in [rand(2) for _ in 1:4], v in [rand(2) for _ in 1:4]
                sop = StatefulJacobianOperator(jac_op, u, prob.p)
                @test (sop * v)≈analytic_jvp(v, u, prob.p) atol=1e-5
                @test (sop' * v)≈analytic_vjp(v, u, prob.p) atol=1e-5

                normal_form_sop = sop' * sop
                JᵀJv = normal_form_sop * v
                J_analytic = analytic_jac(u, prob.p)
                JᵀJv_analytic = J_analytic' * J_analytic * v
                @test JᵀJv≈JᵀJv_analytic atol=1e-5
            end
        end
    end

    prob = NonlinearProblem(
        NonlinearFunction{true}((du, u, p) -> du .= u .^ 2 .- p .+ u[2] * u[1];
            jvp = analytic_jvp!, vjp = analytic_vjp!), [1.0, 3.0], 2.0)

    @testset "Analytic JVP/VJP" begin
        jac_op = JacobianOperator(prob, [2.0, 3.0], prob.u0)

        @testset for u in [rand(2) for _ in 1:4], v in [rand(2) for _ in 1:4]
            sop = StatefulJacobianOperator(jac_op, u, prob.p)
            @test (sop * v)≈analytic_jvp(v, u, prob.p) atol=1e-5
            @test (sop' * v)≈analytic_vjp(v, u, prob.p) atol=1e-5

            normal_form_sop = sop' * sop
            JᵀJv = normal_form_sop * v
            J_analytic = analytic_jac(u, prob.p)
            JᵀJv_analytic = J_analytic' * J_analytic * v
            @test JᵀJv≈JᵀJv_analytic atol=1e-5
        end
    end

    prob = NonlinearProblem(
        NonlinearFunction{true}((du, u, p) -> du .= u .^ 2 .- p .+ u[2] * u[1];
            jac = analytic_jac!), [1.0, 3.0], 2.0)

    @testset "Analytic Jacobian" begin
        jac_op = JacobianOperator(prob, [2.0, 3.0], prob.u0)

        @testset for u in [rand(2) for _ in 1:4], v in [rand(2) for _ in 1:4]
            sop = StatefulJacobianOperator(jac_op, u, prob.p)
            @test (sop * v)≈analytic_jvp(v, u, prob.p) atol=1e-5
            @test (sop' * v)≈analytic_vjp(v, u, prob.p) atol=1e-5

            normal_form_sop = sop' * sop
            JᵀJv = normal_form_sop * v
            J_analytic = analytic_jac(u, prob.p)
            JᵀJv_analytic = J_analytic' * J_analytic * v
            @test JᵀJv≈JᵀJv_analytic atol=1e-5
        end
    end
end

@testitem "Out-of-place Problems" begin
    using ADTypes, SciMLBase
    using Enzyme, ForwardDiff, FiniteDiff, ReverseDiff, Zygote, Tracker
    using SciMLJacobianOperators

    reverse_ADs = [
        AutoEnzyme(),
        AutoEnzyme(; mode = Enzyme.Reverse),
        AutoZygote(),
        AutoTracker(),
        # AutoReverseDiff(), # FIXME: https://github.com/gdalle/DifferentiationInterface.jl/issues/503
        AutoFiniteDiff()
    ]

    forward_ADs = [
        AutoEnzyme(),
        AutoEnzyme(; mode = Enzyme.Forward),
        AutoForwardDiff(),
        AutoFiniteDiff()
    ]

    prob = NonlinearProblem(
        NonlinearFunction{false}((u, p) -> u .^ 2 .- p .+ u[2] * u[1]), [1.0, 3.0], 2.0)

    analytic_jac(u, p) = [2 * u[1]+u[2] u[1]; u[2] 2 * u[2]+u[1]]
    analytic_jvp(v, u, p) = analytic_jac(u, p) * v
    analytic_vjp(v, u, p) = analytic_jac(u, p)' * v

    @testset "AutoDiff" begin
        @testset for jvp_autodiff in forward_ADs, vjp_autodiff in reverse_ADs
            jac_op = JacobianOperator(prob, [2.0, 3.0], prob.u0; jvp_autodiff, vjp_autodiff)

            @testset for u in [rand(2) for _ in 1:4], v in [rand(2) for _ in 1:4]
                sop = StatefulJacobianOperator(jac_op, u, prob.p)
                @test (sop * v)≈analytic_jvp(v, u, prob.p) atol=1e-5
                @test (sop' * v)≈analytic_vjp(v, u, prob.p) atol=1e-5

                normal_form_sop = sop' * sop
                JᵀJv = normal_form_sop * v
                J_analytic = analytic_jac(u, prob.p)
                JᵀJv_analytic = J_analytic' * J_analytic * v
                @test JᵀJv≈JᵀJv_analytic atol=1e-5
            end
        end
    end

    prob = NonlinearProblem(
        NonlinearFunction{false}((u, p) -> u .^ 2 .- p .+ u[2] * u[1];
            vjp = analytic_vjp, jvp = analytic_jvp), [1.0, 3.0], 2.0)

    @testset "Analytic JVP/VJP" begin
        jac_op = JacobianOperator(prob, [2.0, 3.0], prob.u0)

        @testset for u in [rand(2) for _ in 1:4], v in [rand(2) for _ in 1:4]
            sop = StatefulJacobianOperator(jac_op, u, prob.p)
            @test (sop * v)≈analytic_jvp(v, u, prob.p) atol=1e-5
            @test (sop' * v)≈analytic_vjp(v, u, prob.p) atol=1e-5

            normal_form_sop = sop' * sop
            JᵀJv = normal_form_sop * v
            J_analytic = analytic_jac(u, prob.p)
            JᵀJv_analytic = J_analytic' * J_analytic * v
            @test JᵀJv≈JᵀJv_analytic atol=1e-5
        end
    end

    prob = NonlinearProblem(
        NonlinearFunction{false}((u, p) -> u .^ 2 .- p .+ u[2] * u[1];
            jac = analytic_jac), [1.0, 3.0], 2.0)

    @testset "Analytic Jacobian" begin
        jac_op = JacobianOperator(prob, [2.0, 3.0], prob.u0)

        @testset for u in [rand(2) for _ in 1:4], v in [rand(2) for _ in 1:4]
            sop = StatefulJacobianOperator(jac_op, u, prob.p)
            @test (sop * v)≈analytic_jvp(v, u, prob.p) atol=1e-5
            @test (sop' * v)≈analytic_vjp(v, u, prob.p) atol=1e-5

            normal_form_sop = sop' * sop
            JᵀJv = normal_form_sop * v
            J_analytic = analytic_jac(u, prob.p)
            JᵀJv_analytic = J_analytic' * J_analytic * v
            @test JᵀJv≈JᵀJv_analytic atol=1e-5
        end
    end
end

@testitem "Aqua" begin
    using SciMLJacobianOperators, Aqua

    Aqua.test_all(SciMLJacobianOperators)
end

@testitem "Explicit Imports" begin
    using SciMLJacobianOperators, ExplicitImports

    @test check_no_implicit_imports(SciMLJacobianOperators) === nothing
    @test check_no_stale_explicit_imports(SciMLJacobianOperators) === nothing
    @test check_all_qualified_accesses_via_owners(SciMLJacobianOperators) === nothing
end
