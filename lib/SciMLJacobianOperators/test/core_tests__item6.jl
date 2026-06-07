using SciMLJacobianOperators

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
