using NonlinearSolve
using Enzyme
using Reactant
using SciMLBase
using Test

f(u, p) = u .* u .- p
jac(u, p) = reshape(2 .* u, :, 1) .* Float32[1 0; 0 1]
nonlinear_function = NonlinearFunction(f; jac)
autodiff_nonlinear_function = NonlinearFunction(f)

function solve_newton(u, p)
    return solve(NonlinearProblem(nonlinear_function, u, p), NewtonRaphson())
end

function solve_trust_region(u, p)
    return solve(NonlinearProblem(nonlinear_function, u, p), TrustRegion())
end

function solve_default(u, p)
    return solve(NonlinearProblem(nonlinear_function, u, p))
end

function solve_gauss_newton(u, p)
    return solve(
        NonlinearLeastSquaresProblem(nonlinear_function, u, p), GaussNewton()
    )
end

function solve_autodiff_newton(u, p)
    return solve(
        NonlinearProblem(autodiff_nonlinear_function, u, p), NewtonRaphson()
    )
end

function solve_autodiff_trust_region(u, p)
    return solve(
        NonlinearProblem(autodiff_nonlinear_function, u, p), TrustRegion()
    )
end

function solve_autodiff_default(u, p)
    return solve(NonlinearProblem(autodiff_nonlinear_function, u, p))
end

function solve_autodiff_gauss_newton(u, p)
    return solve(
        NonlinearLeastSquaresProblem(autodiff_nonlinear_function, u, p),
        GaussNewton()
    )
end

u0 = Reactant.to_rarray(Float32[1, 1])
p0 = Reactant.to_rarray(Float32[2])
compiled_newton = Reactant.@compile solve_newton(u0, p0)
compiled_trust_region = Reactant.@compile solve_trust_region(u0, p0)
compiled_default = Reactant.@compile solve_default(u0, p0)
compiled_gauss_newton = Reactant.@compile solve_gauss_newton(u0, p0)
compiled_autodiff_newton = Reactant.@compile solve_autodiff_newton(u0, p0)
compiled_autodiff_trust_region = Reactant.@compile solve_autodiff_trust_region(u0, p0)
compiled_autodiff_default = Reactant.@compile solve_autodiff_default(u0, p0)
compiled_autodiff_gauss_newton = Reactant.@compile solve_autodiff_gauss_newton(u0, p0)


# A polyalgorithm's members keep `autodiff = nothing`; the backend is chosen when each
# member is solved, so the choice is only visible on a directly solved algorithm.
for (compiled, name, uses_enzyme) in (
        (compiled_newton, :NewtonRaphson, false),
        (compiled_trust_region, :TrustRegion, false),
        (compiled_default, nothing, false),
        (compiled_gauss_newton, :GaussNewton, false),
        (compiled_autodiff_newton, :NewtonRaphson, true),
        (compiled_autodiff_trust_region, :TrustRegion, true),
        (compiled_autodiff_default, nothing, false),
        (compiled_autodiff_gauss_newton, :GaussNewton, true),
    )
    sol = compiled(
        Reactant.to_rarray(Float32[1, 1]), Reactant.to_rarray(Float32[2])
    )
    @test sol.u isa Reactant.ConcreteRArray
    @test Array(sol.u) ≈ fill(sqrt(2.0f0), 2)
    @test maximum(abs, Array(sol.resid)) ≤ 1.0f-5
    @test sol.retcode == ReturnCode.Success
    @test SciMLBase.successful_retcode(sol)
    if name === nothing
        @test sol.alg isa NonlinearSolvePolyAlgorithm
    else
        @test sol.alg.name === name
    end
    uses_enzyme && @test sol.alg.autodiff isa AutoEnzyme
    @test sol.prob === nothing
    @test sol.stats === nothing
end


function solve_newton_one_step(u, p)
    return solve(
        NonlinearProblem(nonlinear_function, u, p), NewtonRaphson(); maxiters = 1
    )
end


sol_newton_maxiters = Reactant.@jit solve_newton_one_step(
    Reactant.to_rarray(Float32[1, 1]), Reactant.to_rarray(Float32[2])
)
@test sol_newton_maxiters.retcode == ReturnCode.MaxIters
@test !SciMLBase.successful_retcode(sol_newton_maxiters)

struct CompiledProblemSolve{P, F, A}
    problem_type::P
    f::F
    alg::A
end

function (s::CompiledProblemSolve)(u, p)
    prob = s.problem_type(s.f, u, p)
    return s.alg === nothing ? solve(prob; abstol = 1.0f-5) :
        solve(prob, s.alg; abstol = 1.0f-5)
end

# Not compiled here: the least-squares polyalgorithms (and the default least-squares solve)
# contain a member with a line search, whose initialization calls `norm(x, Inf)`, which
# Reactant's overload scalar-indexes; `RobustMultiNewton` contains trust-region schemes whose
# vector-Jacobian products need a reverse-mode pullback, which DifferentiationInterface's
# Enzyme backend does not route through Reactant.
reactant_solver_cases = (
    (:NewtonRaphson, NonlinearProblem, NewtonRaphson()),
    (:TrustRegion, NonlinearProblem, TrustRegion()),
    (:LevenbergMarquardt, NonlinearProblem, LevenbergMarquardt()),
    (
        :LevenbergMarquardtWithoutGeodesic,
        NonlinearProblem,
        LevenbergMarquardt(; disable_geodesic = Val(true)),
    ),
    (:PseudoTransient, NonlinearProblem, PseudoTransient()),
    (
        :FastShortcutNonlinearPolyalg,
        NonlinearProblem,
        FastShortcutNonlinearPolyalg(Float32; u0_len = 2),
    ),
    (
        :NonlinearSolvePolyAlgorithm,
        NonlinearProblem,
        NonlinearSolvePolyAlgorithm((NewtonRaphson(), TrustRegion())),
    ),
    (:DefaultNonlinearSolve, NonlinearProblem, nothing),
    (:GaussNewton, NonlinearLeastSquaresProblem, GaussNewton()),
    (:LeastSquaresTrustRegion, NonlinearLeastSquaresProblem, TrustRegion()),
    (
        :LeastSquaresLevenbergMarquardt,
        NonlinearLeastSquaresProblem,
        LevenbergMarquardt(),
    ),
)

@testset "Analytical Jacobian: $name" for (name, problem_type, alg) in reactant_solver_cases
    compiled = Reactant.compile(
        CompiledProblemSolve(problem_type, nonlinear_function, alg), (u0, p0)
    )
    sol = compiled(
        Reactant.to_rarray(Float32[1, 1]), Reactant.to_rarray(Float32[2])
    )
    @test sol.retcode == ReturnCode.Success
    @test Array(sol.u) ≈ fill(sqrt(2.0f0), 2)
    @test maximum(abs, Array(sol.resid)) ≤ 1.0f-5
end
