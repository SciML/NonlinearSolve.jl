using NonlinearSolveFirstOrder
include("setup_corerootfindtesting.jl")

using LinearAlgebra, SparseArrays, SparseMatrixColorings, SciMLBase

# The pseudo-transient damping term is (1/α) M. As the iteration converges α⁻¹ → 0, so the
# damping vanishes and the method must land on the true root sqrt(p) regardless of which
# (SPD) mass matrix is used. This lets us check the mass-matrix machinery against the known
# solution while also exercising the diagonal / general-matrix code paths.

@testset "mass_matrix = nothing reproduces identity damping" begin
    u0 = [1.0, 1.0]
    prob = NonlinearProblem{false}(quadratic_f, u0, 2.0)

    sol_default = solve(prob, PseudoTransient(; alpha_initial = 10.0); abstol = 1.0e-10)
    sol_nothing = solve(
        prob, PseudoTransient(; alpha_initial = 10.0, mass_matrix = nothing);
        abstol = 1.0e-10
    )

    @test SciMLBase.successful_retcode(sol_default)
    @test sol_default.u == sol_nothing.u
    @test sol_default.stats.nsteps == sol_nothing.stats.nsteps
end

@testset "diagonal mass matrix" begin
    u0 = [1.0, 1.0]
    prob = NonlinearProblem{false}(quadratic_f, u0, 2.0)

    for M in (Diagonal([1.0, 2.0]), Diagonal([0.5, 5.0]))
        solver = PseudoTransient(; alpha_initial = 10.0, mass_matrix = M)
        sol = solve(prob, solver; abstol = 1.0e-10)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [sqrt(2.0), sqrt(2.0)] atol = 1.0e-7
    end
end

@testset "general (dense) mass matrix" begin
    u0 = [1.0, 1.0]
    prob = NonlinearProblem{false}(quadratic_f, u0, 2.0)

    M = [2.0 0.5; 0.5 2.0]  # SPD, non-diagonal
    solver = PseudoTransient(; alpha_initial = 10.0, mass_matrix = M)
    sol = solve(prob, solver; abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [sqrt(2.0), sqrt(2.0)] atol = 1.0e-7
end

@testset "in-place with mass matrix" begin
    u0 = [1.0, 1.0]
    prob = NonlinearProblem{true}(quadratic_f!, u0, 2.0)

    solver = PseudoTransient(; alpha_initial = 10.0, mass_matrix = Diagonal([1.0, 3.0]))
    sol = solve(prob, solver; abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ [sqrt(2.0), sqrt(2.0)] atol = 1.0e-7
end

# The mass matrix should also be picked up automatically from the problem's
# `NonlinearFunction` when one is present, so DAE-derived `NonlinearProblem`s work with no
# extra keyword.
@testset "mass matrix from NonlinearFunction" begin
    if hasproperty(NonlinearFunction(quadratic_f), :mass_matrix)
        M = Diagonal([1.0, 2.0])
        nlf = NonlinearFunction{false}(quadratic_f; mass_matrix = M)
        prob = NonlinearProblem(nlf, [1.0, 1.0], 2.0)

        # Explicit keyword omitted -> resolved from the function's mass matrix.
        sol_auto = solve(prob, PseudoTransient(; alpha_initial = 10.0); abstol = 1.0e-10)
        @test SciMLBase.successful_retcode(sol_auto)
        @test sol_auto.u ≈ [sqrt(2.0), sqrt(2.0)] atol = 1.0e-7

        # Explicit keyword should match the automatically-resolved behavior.
        sol_explicit = solve(
            NonlinearProblem{false}(quadratic_f, [1.0, 1.0], 2.0),
            PseudoTransient(; alpha_initial = 10.0, mass_matrix = M); abstol = 1.0e-10
        )
        @test sol_auto.u ≈ sol_explicit.u
    end
end

# Singular mass matrix: the headline use case (semi-explicit index-1 DAE steady state). The
# algebraic component has zero mass, so its row is treated as pure Newton, the differential
# component is damped. Must still converge.
@testset "singular (DAE) mass matrix" begin
    # u1 differential (mass 1), u2 algebraic (mass 0)
    f(u, p) = [u[1]^2 - u[2] - 1.0, u[1] + u[2] - 3.0]
    xstar = (-1 + sqrt(17)) / 2
    ustar = [xstar, 3 - xstar]

    prob = NonlinearProblem{false}(f, [3.0, -2.0], nothing)
    for M in (Diagonal([1.0, 0.0]), [1.0 0.0; 0.0 0.0])
        sol = solve(prob, PseudoTransient(; alpha_initial = 1.0e-2, mass_matrix = M); abstol = 1.0e-10)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ ustar atol = 1.0e-7
    end
end

# Explicit `mass_matrix = I` / `λI` must not error (regression for the UniformScaling path).
@testset "UniformScaling mass matrix" begin
    prob = NonlinearProblem{false}(quadratic_f, [1.0, 1.0], 2.0)

    # `I` is identity damping: must match the default exactly.
    s_default = solve(prob, PseudoTransient(; alpha_initial = 10.0); abstol = 1.0e-10)
    s_I = solve(prob, PseudoTransient(; alpha_initial = 10.0, mass_matrix = I); abstol = 1.0e-10)
    @test s_I.u == s_default.u
    @test s_I.stats.nsteps == s_default.stats.nsteps

    # A scaled identity must still converge.
    s_2I = solve(prob, PseudoTransient(; alpha_initial = 10.0, mass_matrix = 2I); abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(s_2I)
    @test s_2I.u ≈ [sqrt(2.0), sqrt(2.0)] atol = 1.0e-7
end

# Sparse in-place Jacobian must not have its sparsity pattern corrupted by the damping term.
@testset "sparse Jacobian is not corrupted by mass-matrix damping" begin
    quad!(du, u, p) = (du .= u .* u .- p)
    jp = sparse(Diagonal(ones(2)))
    nlf = NonlinearFunction{true}(quad!; jac_prototype = jp)
    prob = NonlinearProblem(nlf, [1.0, 1.0], 2.0)

    # Off-diagonal mass matrix entries lie outside the Jacobian pattern.
    for M in (sparse([2.0 0.5; 0.5 3.0]), [2.0 0.5; 0.5 3.0])
        sol = solve(prob, PseudoTransient(; alpha_initial = 10.0, mass_matrix = M); abstol = 1.0e-9)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ [sqrt(2.0), sqrt(2.0)] atol = 1.0e-6
    end
end

# A mass matrix whose size does not match the number of unknowns must error clearly rather
# than reading out of bounds.
@testset "size-mismatched mass matrix errors" begin
    prob = NonlinearProblem{false}(quadratic_f, [1.0, 1.0, 1.0], 2.0)
    @test_throws DimensionMismatch solve(
        prob, PseudoTransient(; alpha_initial = 10.0, mass_matrix = Diagonal([1.0, 2.0]));
        abstol = 1.0e-10
    )
end
