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
    @test sol.retcode isa Reactant.ConcreteEnum{ReturnCode.T}
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
@test sol_newton_maxiters.retcode isa Reactant.ConcreteEnum{ReturnCode.T}
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
    (:SmallRadiusTrustRegion, NonlinearProblem, TrustRegion(; initial_trust_radius = 0.05f0)),
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
    @test sol.retcode isa Reactant.ConcreteEnum{ReturnCode.T}
    @test sol.retcode == ReturnCode.Success
    @test Array(sol.u) ≈ fill(sqrt(2.0f0), 2)
    @test maximum(abs, Array(sol.resid)) ≤ 1.0f-5
end

@testset "Moré runtime branches" begin
    solver = CompiledProblemSolve(
        NonlinearProblem, nonlinear_function,
        TrustRegion(; initial_trust_radius = 0.05f0)
    )
    compiled = Reactant.compile(solver, (u0, p0))
    for target in (1.0f0, 2.0f0, 10.0f0)
        sol = compiled(Reactant.to_rarray(Float32[1, 1]), Reactant.to_rarray(Float32[target]))
        @test sol.retcode isa Reactant.ConcreteEnum{ReturnCode.T}
        @test sol.retcode == ReturnCode.Success
        @test maximum(abs, Array(sol.resid)) ≤ 1.0f-5
    end
end

# Quasi-Newton families with runtime (non-constant) inputs, compared to host.
# DFSane is rejected under compile (see dedicated test below).
runtime_quasi_newton_cases = (
    (:Broyden, Broyden()),
    (:Klement, Klement()),
    (:LimitedMemoryBroyden, LimitedMemoryBroyden(; threshold = 2)),
)

@testset "Runtime inputs: $name" for (name, alg) in runtime_quasi_newton_cases
    function dosolve(u, p)
        return solve(NonlinearProblem(f, u, p), alg; maxiters = 50, abstol = 1.0f-5)
    end
    u_host = Float32[1, 1]
    p_host = Float32[2]
    sol_host = dosolve(u_host, p_host)
    sol = Reactant.@jit dosolve(Reactant.to_rarray(u_host), Reactant.to_rarray(p_host))
    @test sol.retcode isa Reactant.ConcreteEnum{ReturnCode.T}
    @test sol.retcode == ReturnCode.Success
    @test Array(sol.u) ≈ Array(sol_host.u) rtol = 1.0f-3
    @test maximum(abs, Array(sol.resid)) ≤ 1.0f-4
end

@testset "DFSane is rejected under compile" begin
    function dosolve_dfsane(u, p)
        return solve(NonlinearProblem(f, u, p), DFSane(); maxiters = 50, abstol = 1.0f-5)
    end
    err = try
        Reactant.@jit dosolve_dfsane(
            Reactant.to_rarray(Float32[1, 1]), Reactant.to_rarray(Float32[2])
        )
        nothing
    catch e
        e
    end
    @test err isa Exception
    @test occursin("DFSane", sprint(showerror, err))
    @test occursin("line search", sprint(showerror, err))
end

# Non-finite residual must not report Success (Reactant `maximum` drops NaN).
@testset "NaN residual retcode: $name" for (name, alg) in (
        (:NewtonRaphson, NewtonRaphson()),
        (:Broyden, Broyden()),
        (:LimitedMemoryBroyden, LimitedMemoryBroyden(; threshold = 2)),
    )
    function dosolve_nan(u, p)
        return solve(NonlinearProblem(f, u, p), alg; maxiters = 50, abstol = 1.0f-5)
    end
    u_host = Float32[1, 1]
    p_host = Float32[NaN]
    sol_host = dosolve_nan(u_host, p_host)
    sol = Reactant.@jit dosolve_nan(Reactant.to_rarray(u_host), Reactant.to_rarray(p_host))
    @test sol.retcode isa Reactant.ConcreteEnum{ReturnCode.T}
    @test sol.retcode == sol_host.retcode
    @test sol.retcode == ReturnCode.Unstable
end

@testset "LimitedMemoryBroyden cube retcode matches host" begin
    fcube(u, p) = u .^ 3 .- p
    function dosolve_cube(u, p)
        return solve(
            NonlinearProblem(fcube, u, p), LimitedMemoryBroyden(; threshold = 2);
            maxiters = 50, abstol = 1.0f-5
        )
    end
    u_host = Float32[3, -2]
    p_host = Float32[2]
    sol_host = dosolve_cube(u_host, p_host)
    sol = Reactant.@jit dosolve_cube(Reactant.to_rarray(u_host), Reactant.to_rarray(p_host))
    @test sol.retcode isa Reactant.ConcreteEnum{ReturnCode.T}
    @test sol.retcode == sol_host.retcode
end

@testset "Quasi-Newton max_resets under compile: $name" for (name, f, u0, alg) in (
        (
            :LimitedMemoryBroyden,
            (u, p) -> exp.(u) .- p,
            Float32[3, 4],
            LimitedMemoryBroyden(; max_resets = 1),
        ),
        (
            :Klement,
            (u, p) -> u .^ 3 .- p,
            Float32[3, -2],
            Klement(; max_resets = 1),
        ),
    )
    function dosolve_reset(u, p)
        return solve(NonlinearProblem(f, u, p), alg; maxiters = 50, abstol = 1.0f-5)
    end
    p_host = Float32[2]
    sol_host = dosolve_reset(u0, p_host)
    sol = Reactant.@jit dosolve_reset(Reactant.to_rarray(u0), Reactant.to_rarray(p_host))
    @test sol.retcode isa Reactant.ConcreteEnum{ReturnCode.T}
    @test sol.retcode == sol_host.retcode
    @test sol.retcode == ReturnCode.ConvergenceFailure
end

@testset "Polyalgorithm store_original under compile" begin
    function polyrun(u, p)
        return solve(
            NonlinearProblem(nonlinear_function, u, p),
            NonlinearSolvePolyAlgorithm((NewtonRaphson(),); store_original = Val(true));
            maxiters = 50,
            alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = true)
        )
    end
    sol_host = polyrun(Float32[1, 1], Float32[2])
    @test !isnothing(sol_host.original)
    sol = Reactant.@jit polyrun(
        Reactant.to_rarray(Float32[1, 1]), Reactant.to_rarray(Float32[2])
    )
    @test Array(sol.u) ≈ Array(sol_host.u) rtol = 1.0f-4
    @test !isnothing(sol.original)
end
