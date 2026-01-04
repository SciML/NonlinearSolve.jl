@testitem "Scalar Ops" begin
    using ADTypes, SciMLBase
    using Zygote, ForwardDiff, FiniteDiff, ReverseDiff, Tracker
    using SciMLJacobianOperators

    # Conditionally import Enzyme only if not on Julia prerelease
    if isempty(VERSION.prerelease) && VERSION < v"1.12"
        using Enzyme
    end

    reverse_ADs = [
        AutoZygote(),
        AutoReverseDiff(),
        AutoTracker(),
        AutoFiniteDiff(),
    ]
    if isempty(VERSION.prerelease) && VERSION < v"1.12"
        push!(reverse_ADs, AutoEnzyme())
        push!(reverse_ADs, AutoEnzyme(; mode = Enzyme.Reverse))
    end

    forward_ADs = [
        AutoForwardDiff(),
        AutoFiniteDiff(),
    ]
    if isempty(VERSION.prerelease) && VERSION < v"1.12"
        push!(forward_ADs, AutoEnzyme())
        push!(forward_ADs, AutoEnzyme(; mode = Enzyme.Forward))
    end

    prob = NonlinearProblem(NonlinearFunction{false}((u, p) -> u^2 - p), 1.0, 2.0)

    analytic_jac(u, p) = 2 * u
    analytic_jvp(v, u, p) = 2 * u * v
    analytic_vjp(v, u, p) = analytic_jvp(v, u, p)

    @testset "AutoDiff" begin
        @testset for jvp_autodiff in forward_ADs, vjp_autodiff in reverse_ADs

            jac_op = JacobianOperator(prob, -1.0, 1.0; jvp_autodiff, vjp_autodiff)

            @testset for u in rand(4), v in rand(4)

                sop = StatefulJacobianOperator(jac_op, u, prob.p)
                @test (sop * v) ≈ analytic_jvp(v, u, prob.p) atol = 1.0e-5
                @test (sop' * v) ≈ analytic_vjp(v, u, prob.p) atol = 1.0e-5

                normal_form_sop = sop' * sop
                JᵀJv = normal_form_sop * v
                J_analytic = analytic_jac(u, prob.p)
                JᵀJv_analytic = J_analytic' * J_analytic * v
                @test JᵀJv ≈ JᵀJv_analytic atol = 1.0e-5
            end
        end
    end

    prob = NonlinearProblem(
        NonlinearFunction{false}((u, p) -> u^2 - p; jvp = analytic_jvp, vjp = analytic_vjp),
        1.0, 2.0
    )

    @testset "Analytic JVP/VJP" begin
        jac_op = JacobianOperator(prob, -1.0, 1.0)

        @testset for u in rand(4), v in rand(4)

            sop = StatefulJacobianOperator(jac_op, u, prob.p)
            @test (sop * v) ≈ analytic_jvp(v, u, prob.p) atol = 1.0e-5
            @test (sop' * v) ≈ analytic_vjp(v, u, prob.p) atol = 1.0e-5

            normal_form_sop = sop' * sop
            JᵀJv = normal_form_sop * v
            J_analytic = analytic_jac(u, prob.p)
            JᵀJv_analytic = J_analytic' * J_analytic * v
            @test JᵀJv ≈ JᵀJv_analytic atol = 1.0e-5
        end
    end

    prob = NonlinearProblem(
        NonlinearFunction{false}((u, p) -> u^2 - p; jac = (u, p) -> 2 * u), 1.0, 2.0
    )

    @testset "Analytic Jacobian" begin
        jac_op = JacobianOperator(prob, -1.0, 1.0)

        @testset for u in rand(4), v in rand(4)

            sop = StatefulJacobianOperator(jac_op, u, prob.p)
            @test (sop * v) ≈ 2 * u * v atol = 1.0e-5
            @test (sop' * v) ≈ 2 * u * v atol = 1.0e-5

            normal_form_sop = sop' * sop
            JᵀJv = normal_form_sop * v
            J_analytic = analytic_jac(u, prob.p)
            JᵀJv_analytic = J_analytic' * J_analytic * v
            @test JᵀJv ≈ JᵀJv_analytic atol = 1.0e-5
        end
    end
end

@testitem "Inplace Problems" begin
    using ADTypes, SciMLBase
    using ForwardDiff, FiniteDiff, ReverseDiff
    using SciMLJacobianOperators

    # Conditionally import Enzyme only if not on Julia prerelease
    if isempty(VERSION.prerelease) && VERSION < v"1.12"
        using Enzyme
    end

    reverse_ADs = [
        AutoReverseDiff(),
        AutoFiniteDiff(),
    ]
    if isempty(VERSION.prerelease) && VERSION < v"1.12"
        push!(reverse_ADs, AutoEnzyme())
        push!(reverse_ADs, AutoEnzyme(; mode = Enzyme.Reverse))
    end

    forward_ADs = [
        AutoForwardDiff(),
        AutoFiniteDiff(),
    ]
    if isempty(VERSION.prerelease) && VERSION < v"1.12"
        push!(forward_ADs, AutoEnzyme())
        push!(forward_ADs, AutoEnzyme(; mode = Enzyme.Forward))
    end

    prob = NonlinearProblem(
        NonlinearFunction{true}((du, u, p) -> du .= u .^ 2 .- p .+ u[2] * u[1]), [1.0, 3.0], 2.0
    )

    analytic_jac(u, p) = [2 * u[1] + u[2] u[1]; u[2] 2 * u[2] + u[1]]
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
                @test (sop * v) ≈ analytic_jvp(v, u, prob.p) atol = 1.0e-5
                @test (sop' * v) ≈ analytic_vjp(v, u, prob.p) atol = 1.0e-5

                normal_form_sop = sop' * sop
                JᵀJv = normal_form_sop * v
                J_analytic = analytic_jac(u, prob.p)
                JᵀJv_analytic = J_analytic' * J_analytic * v
                @test JᵀJv ≈ JᵀJv_analytic atol = 1.0e-5
            end
        end
    end

    prob = NonlinearProblem(
        NonlinearFunction{true}(
            (du, u, p) -> du .= u .^ 2 .- p .+ u[2] * u[1];
            jvp = analytic_jvp!, vjp = analytic_vjp!
        ), [1.0, 3.0], 2.0
    )

    @testset "Analytic JVP/VJP" begin
        jac_op = JacobianOperator(prob, [2.0, 3.0], prob.u0)

        @testset for u in [rand(2) for _ in 1:4], v in [rand(2) for _ in 1:4]

            sop = StatefulJacobianOperator(jac_op, u, prob.p)
            @test (sop * v) ≈ analytic_jvp(v, u, prob.p) atol = 1.0e-5
            @test (sop' * v) ≈ analytic_vjp(v, u, prob.p) atol = 1.0e-5

            normal_form_sop = sop' * sop
            JᵀJv = normal_form_sop * v
            J_analytic = analytic_jac(u, prob.p)
            JᵀJv_analytic = J_analytic' * J_analytic * v
            @test JᵀJv ≈ JᵀJv_analytic atol = 1.0e-5
        end
    end

    prob = NonlinearProblem(
        NonlinearFunction{true}(
            (du, u, p) -> du .= u .^ 2 .- p .+ u[2] * u[1];
            jac = analytic_jac!
        ), [1.0, 3.0], 2.0
    )

    @testset "Analytic Jacobian" begin
        jac_op = JacobianOperator(prob, [2.0, 3.0], prob.u0)

        @testset for u in [rand(2) for _ in 1:4], v in [rand(2) for _ in 1:4]

            sop = StatefulJacobianOperator(jac_op, u, prob.p)
            @test (sop * v) ≈ analytic_jvp(v, u, prob.p) atol = 1.0e-5
            @test (sop' * v) ≈ analytic_vjp(v, u, prob.p) atol = 1.0e-5

            normal_form_sop = sop' * sop
            JᵀJv = normal_form_sop * v
            J_analytic = analytic_jac(u, prob.p)
            JᵀJv_analytic = J_analytic' * J_analytic * v
            @test JᵀJv ≈ JᵀJv_analytic atol = 1.0e-5
        end
    end
end

@testitem "Out-of-place Problems" begin
    using ADTypes, SciMLBase
    using ForwardDiff, FiniteDiff, ReverseDiff, Zygote, Tracker
    using SciMLJacobianOperators

    # Conditionally import Enzyme only if not on Julia prerelease
    if isempty(VERSION.prerelease) && VERSION < v"1.12"
        using Enzyme
    end

    reverse_ADs = [
        AutoZygote(),
        AutoTracker(),
        AutoReverseDiff(),
        AutoFiniteDiff(),
    ]
    if isempty(VERSION.prerelease) && VERSION < v"1.12"
        push!(reverse_ADs, AutoEnzyme())
        push!(reverse_ADs, AutoEnzyme(; mode = Enzyme.Reverse))
    end

    forward_ADs = [
        AutoForwardDiff(),
        AutoFiniteDiff(),
    ]
    if isempty(VERSION.prerelease) && VERSION < v"1.12"
        push!(forward_ADs, AutoEnzyme())
        push!(forward_ADs, AutoEnzyme(; mode = Enzyme.Forward))
    end

    prob = NonlinearProblem(
        NonlinearFunction{false}((u, p) -> u .^ 2 .- p .+ u[2] * u[1]), [1.0, 3.0], 2.0
    )

    analytic_jac(u, p) = [2 * u[1] + u[2] u[1]; u[2] 2 * u[2] + u[1]]
    analytic_jvp(v, u, p) = analytic_jac(u, p) * v
    analytic_vjp(v, u, p) = analytic_jac(u, p)' * v

    @testset "AutoDiff" begin
        @testset for jvp_autodiff in forward_ADs, vjp_autodiff in reverse_ADs

            jac_op = JacobianOperator(prob, [2.0, 3.0], prob.u0; jvp_autodiff, vjp_autodiff)

            @testset for u in [rand(2) for _ in 1:4], v in [rand(2) for _ in 1:4]

                sop = StatefulJacobianOperator(jac_op, u, prob.p)
                @test (sop * v) ≈ analytic_jvp(v, u, prob.p) atol = 1.0e-5
                @test (sop' * v) ≈ analytic_vjp(v, u, prob.p) atol = 1.0e-5

                normal_form_sop = sop' * sop
                JᵀJv = normal_form_sop * v
                J_analytic = analytic_jac(u, prob.p)
                JᵀJv_analytic = J_analytic' * J_analytic * v
                @test JᵀJv ≈ JᵀJv_analytic atol = 1.0e-5
            end
        end
    end

    prob = NonlinearProblem(
        NonlinearFunction{false}(
            (u, p) -> u .^ 2 .- p .+ u[2] * u[1];
            vjp = analytic_vjp, jvp = analytic_jvp
        ), [1.0, 3.0], 2.0
    )

    @testset "Analytic JVP/VJP" begin
        jac_op = JacobianOperator(prob, [2.0, 3.0], prob.u0)

        @testset for u in [rand(2) for _ in 1:4], v in [rand(2) for _ in 1:4]

            sop = StatefulJacobianOperator(jac_op, u, prob.p)
            @test (sop * v) ≈ analytic_jvp(v, u, prob.p) atol = 1.0e-5
            @test (sop' * v) ≈ analytic_vjp(v, u, prob.p) atol = 1.0e-5

            normal_form_sop = sop' * sop
            JᵀJv = normal_form_sop * v
            J_analytic = analytic_jac(u, prob.p)
            JᵀJv_analytic = J_analytic' * J_analytic * v
            @test JᵀJv ≈ JᵀJv_analytic atol = 1.0e-5
        end
    end

    prob = NonlinearProblem(
        NonlinearFunction{false}(
            (u, p) -> u .^ 2 .- p .+ u[2] * u[1];
            jac = analytic_jac
        ), [1.0, 3.0], 2.0
    )

    @testset "Analytic Jacobian" begin
        jac_op = JacobianOperator(prob, [2.0, 3.0], prob.u0)

        @testset for u in [rand(2) for _ in 1:4], v in [rand(2) for _ in 1:4]

            sop = StatefulJacobianOperator(jac_op, u, prob.p)
            @test (sop * v) ≈ analytic_jvp(v, u, prob.p) atol = 1.0e-5
            @test (sop' * v) ≈ analytic_vjp(v, u, prob.p) atol = 1.0e-5

            normal_form_sop = sop' * sop
            JᵀJv = normal_form_sop * v
            J_analytic = analytic_jac(u, prob.p)
            JᵀJv_analytic = J_analytic' * J_analytic * v
            @test JᵀJv ≈ JᵀJv_analytic atol = 1.0e-5
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

@testitem "Copy with Tuple parameters (Issue #752)" begin
    using ADTypes, SciMLBase, ForwardDiff, FiniteDiff
    using SciMLJacobianOperators
    using SciMLJacobianOperators: _safe_copy

    # Test _safe_copy helper functions
    @testset "_safe_copy helpers" begin
        # Tuples should be returned as-is (immutable)
        t = (1.0, 2.0, 3.0, 4.0)
        @test _safe_copy(t) === t

        # NamedTuples should be returned as-is (immutable)
        nt = (a = 1.0, b = 2.0)
        @test _safe_copy(nt) === nt

        # Numbers should be returned as-is (immutable)
        @test _safe_copy(42.0) === 42.0
        @test _safe_copy(42) === 42

        # Arrays should be copied
        arr = [1.0, 2.0, 3.0]
        arr_copy = _safe_copy(arr)
        @test arr_copy == arr
        @test arr_copy !== arr  # Different object
    end

    # Test copy(StatefulJacobianOperator) with Tuple parameter
    # This is a regression test for issue #752 where JFNK failed with:
    # MethodError: no method matching copy(::NTuple{4, Float64})
    @testset "copy(StatefulJacobianOperator) with Tuple p" begin
        # Brusselator-like problem setup with Tuple parameters
        function brusselator_f(du, u, p)
            α, β, γ, δ = p
            du[1] = α + u[1]^2 * u[2] - (β + 1) * u[1]
            du[2] = β * u[1] - u[1]^2 * u[2]
            return nothing
        end

        # Parameters as a Tuple (this is what caused issue #752)
        p_tuple = (1.0, 3.0, 1.0, 1.0)
        u0 = [1.0, 1.0]
        fu0 = similar(u0)
        brusselator_f(fu0, u0, p_tuple)

        prob = NonlinearProblem(
            NonlinearFunction{true}(brusselator_f), u0, p_tuple
        )

        jac_op = JacobianOperator(
            prob, fu0, u0; jvp_autodiff = AutoForwardDiff(), vjp_autodiff = AutoFiniteDiff()
        )
        sop = StatefulJacobianOperator(jac_op, u0, p_tuple)

        # This should not throw MethodError: no method matching copy(::NTuple{4, Float64})
        sop_copy = copy(sop)

        @test sop_copy.p === p_tuple  # Same tuple (immutable, no copy needed)
        @test sop_copy.u == u0
        @test sop_copy.u !== sop.u  # Different array object (copied)

        # Verify the copied operator still works
        v = [1.0, 0.0]
        result_original = sop * v
        result_copy = sop_copy * v
        @test result_original ≈ result_copy
    end

    # Test copy(StatefulJacobianOperator) with NamedTuple parameter
    @testset "copy(StatefulJacobianOperator) with NamedTuple p" begin
        function simple_f(du, u, p)
            du[1] = p.a * u[1] + p.b * u[2]
            du[2] = p.c * u[1] + p.d * u[2]
            return nothing
        end

        p_namedtuple = (a = 1.0, b = 2.0, c = 3.0, d = 4.0)
        u0 = [1.0, 1.0]
        fu0 = similar(u0)
        simple_f(fu0, u0, p_namedtuple)

        prob = NonlinearProblem(
            NonlinearFunction{true}(simple_f), u0, p_namedtuple
        )

        jac_op = JacobianOperator(
            prob, fu0, u0; jvp_autodiff = AutoForwardDiff(), vjp_autodiff = AutoFiniteDiff()
        )
        sop = StatefulJacobianOperator(jac_op, u0, p_namedtuple)

        # This should not throw
        sop_copy = copy(sop)

        @test sop_copy.p === p_namedtuple  # Same NamedTuple (immutable)
    end
end
