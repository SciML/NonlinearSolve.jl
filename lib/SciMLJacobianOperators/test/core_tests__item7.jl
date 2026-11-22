using SciMLJacobianOperators

using ADTypes, SciMLBase, LinearAlgebra
using ForwardDiff
using RecursiveArrayTools: ArrayPartition
using StaticArrays: @SVector, @MVector, SVector
using Zygote

# Conditionally import Enzyme only if not on Julia prerelease / < 1.12
enzyme_ok = isempty(VERSION.prerelease) && VERSION < v"1.12"
if enzyme_ok
    using Enzyme
end

f_oop(u, p) = u .* u .- p
function f_iip(du, u, p)
    du .= u .* u .- p
    return nothing
end

analytic_jvp(v, u, p) = 2 .* collect(u) .* v
analytic_vjp(v, u, p) = 2 .* collect(u) .* v

if enzyme_ok
    @testset "SVector OOP + dense Vector seed (Enzyme)" begin
        u0 = @SVector [1.0, 1.0]
        fu0 = f_oop(u0, 2.0)
        prob = NonlinearProblem{false}(f_oop, u0, 2.0)
        jac_op = JacobianOperator(
            prob, fu0, u0;
            jvp_autodiff = AutoForwardDiff(), vjp_autodiff = AutoEnzyme()
        )
        sop = StatefulJacobianOperator(jac_op, u0, 2.0)
        v = [0.3, 0.7]
        @test collect(sop * v) ≈ analytic_jvp(v, u0, 2.0) atol = 1.0e-8
        @test collect(sop' * v) ≈ analytic_vjp(v, u0, 2.0) atol = 1.0e-8
        Jv = similar(v)
        vJ = similar(v)
        mul!(Jv, sop, v)
        mul!(vJ, sop', v)
        @test Jv ≈ analytic_jvp(v, u0, 2.0) atol = 1.0e-8
        @test vJ ≈ analytic_vjp(v, u0, 2.0) atol = 1.0e-8
    end

    @testset "MVector IIP + dense Vector buffers (Enzyme)" begin
        u0 = @MVector [1.0, 1.0]
        fu0 = @MVector [0.0, 0.0]
        f_iip(fu0, u0, 2.0)
        prob = NonlinearProblem{true}(f_iip, u0, 2.0)
        jac_op = JacobianOperator(
            prob, fu0, u0;
            jvp_autodiff = AutoEnzyme(; mode = Enzyme.Forward),
            vjp_autodiff = AutoEnzyme(; mode = Enzyme.Reverse)
        )
        sop = StatefulJacobianOperator(jac_op, u0, 2.0)
        v = [0.3, 0.7]
        Jv = similar(v)
        vJ = similar(v)
        mul!(Jv, sop, v)
        mul!(vJ, sop', v)
        @test Jv ≈ analytic_jvp(v, u0, 2.0) atol = 1.0e-8
        @test vJ ≈ analytic_vjp(v, u0, 2.0) atol = 1.0e-8
    end
end

@testset "ArrayPartition (no convert) + ForwardDiff" begin
    # Types without `convert(T, ::AbstractArray)` must keep working — `_shaped_like`
    # must not force convert on every primal.
    u0 = ArrayPartition([1.0], [2.0])
    fu0 = f_oop(u0, 2.0)
    prob = NonlinearProblem{false}(f_oop, u0, 2.0)
    jac_op = JacobianOperator(
        prob, fu0, u0;
        jvp_autodiff = AutoForwardDiff(), vjp_autodiff = AutoForwardDiff()
    )
    sop = StatefulJacobianOperator(jac_op, u0, 2.0)
    v = [0.3, 0.7]
    @test collect(sop * v) ≈ analytic_jvp(v, u0, 2.0) atol = 1.0e-8
    @test collect(sop' * v) ≈ analytic_vjp(v, u0, 2.0) atol = 1.0e-8
end

@testset "Vector←Vector seed is a no-op reshape" begin
    u0 = [1.0, 1.0]
    v = [0.3, 0.7]
    shaped = SciMLJacobianOperators._shaped_like(u0, v)
    @test shaped isa Vector{Float64}
    # Hot path: reshape shares data with the input Vector (no convert / copy).
    @test pointer(shaped) == pointer(v)
end

@testset "Preserve seed scalar type (ForwardDiff/Zygote)" begin
    # Coercion must not convert seeds to eltype(u): Float32 static state plus a
    # large/precise Float64 seed, and nested ForwardDiff through `sop * v`.
    f_sq(u, p) = u .* u
    u32 = SVector(1.0f0, 2.0f0)
    fu32 = f_sq(u32, nothing)
    sop32 = StatefulJacobianOperator(
        JacobianOperator(
            NonlinearProblem{false}(f_sq, u32), fu32, u32;
            jvp_autodiff = AutoForwardDiff(), vjp_autodiff = AutoZygote()
        ),
        u32,
        nothing
    )
    for v in ([1.0 + 2.0^-30, 2.0 + 2.0^-30], [1.0e40, 2.0e40])
        expected = 2 .* Float64.(u32) .* v
        @test sop32 * v == expected
        @test sop32' * v == expected
    end
    u64 = SVector(1.0, 2.0)
    fu64 = f_sq(u64, nothing)
    sop64 = StatefulJacobianOperator(
        JacobianOperator(
            NonlinearProblem{false}(f_sq, u64), fu64, u64;
            jvp_autodiff = AutoForwardDiff(), vjp_autodiff = AutoZygote()
        ),
        u64,
        nothing
    )
    @test ForwardDiff.jacobian(v -> sop64 * v, [0.3, 0.7]) == [2.0 0.0; 0.0 4.0]
end
