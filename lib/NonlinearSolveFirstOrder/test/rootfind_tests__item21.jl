using NonlinearSolveFirstOrder
include("setup_corerootfindtesting.jl")

using LinearAlgebra, SciMLBase

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
