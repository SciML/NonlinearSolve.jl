using NonlinearSolve
using DifferentiationInterface
using Enzyme
using LinearAlgebra
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

# Analytical-Jacobian compiles (no DifferentiationInterface Jacobian path).
# A polyalgorithm's members keep `autodiff = nothing`; the backend is chosen when each
# member is solved, so the choice is only visible on a directly solved algorithm.
for (compiled, name, uses_enzyme) in (
        (compiled_newton, :NewtonRaphson, false),
        (compiled_trust_region, :TrustRegion, false),
        (compiled_default, nothing, false),
        (compiled_gauss_newton, :GaussNewton, false),
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
    @test sol.prob === nothing
    @test sol.stats === nothing
end

# Autodiff array Jacobians under `@compile` require an analytic `jac`: registered
# DifferentiationInterface cannot yet build array Jacobians on traced arrays
# (JuliaDiff/DifferentiationInterface.jl#1067). Throw before DI prepare so
# `@compile` fails loudly instead of hanging on scalar indexing.
@testset "Autodiff Jacobian under compile requires analytic jac: $name" for (solve_fn, name) in (
        (solve_autodiff_newton, :NewtonRaphson),
        (solve_autodiff_trust_region, :TrustRegion),
        (solve_autodiff_default, nothing),
        (solve_autodiff_gauss_newton, :GaussNewton),
    )
    err = try
        Reactant.@compile solve_fn(u0, p0)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("analytic", sprint(showerror, err))
    @test occursin("DifferentiationInterface", sprint(showerror, err))
    @test occursin("Krylov", sprint(showerror, err))
end

@testset "Broyden true_jacobian without analytic jac under compile" begin
    function solve_broyden_true_jac_no_analytic(u, p)
        return solve(
            NonlinearProblem(NonlinearFunction(f), u, p),
            Broyden(; init_jacobian = Val(:true_jacobian))
        )
    end
    err = try
        Reactant.@compile solve_broyden_true_jac_no_analytic(u0, p0)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("analytic", sprint(showerror, err))
end

# Scalar AD derivatives under Reactant compile use DI.derivative (not the array
# Jacobian path) and select AutoForwardFromPrimitive(AutoEnzyme Forward).
fs(u, p) = u * u - p
function solve_scalar_newton(u, p)
    return solve(NonlinearProblem(NonlinearFunction(fs), u, p), NewtonRaphson())
end
function solve_scalar_default(u, p)
    return solve(NonlinearProblem(NonlinearFunction(fs), u, p))
end

@testset "Scalar autodiff Newton / default under compile match host" begin
    u_host = 1.0f0
    p_host = 2.0f0
    ru = Reactant.ConcreteRNumber(u_host)
    rp = Reactant.ConcreteRNumber(p_host)
    for (solve_fn, check_backend) in (
            (solve_scalar_newton, true),
            (solve_scalar_default, false),
        )
        sol_host = solve_fn(u_host, p_host)
        compiled = Reactant.@compile solve_fn(ru, rp)
        sol = compiled(Reactant.ConcreteRNumber(u_host), Reactant.ConcreteRNumber(p_host))
        @test sol.retcode isa Reactant.ConcreteEnum{ReturnCode.T}
        @test sol.retcode == ReturnCode.Success
        @test sol.retcode == sol_host.retcode
        @test Float32(sol.u) ≈ Float32(sol_host.u) rtol = 1.0f-5
        if check_backend
            @test sol.alg.autodiff isa DifferentiationInterface.AutoForwardFromPrimitive
        end
    end
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
# Newton needs an analytic `jac` under compile (AD Jacobians throw; see DI#1067).
@testset "NaN residual retcode: $name" for (name, alg, nf) in (
        (:NewtonRaphson, NewtonRaphson(), nonlinear_function),
        (:Broyden, Broyden(), NonlinearFunction(f)),
        (:LimitedMemoryBroyden, LimitedMemoryBroyden(; threshold = 2), NonlinearFunction(f)),
    )
    function dosolve_nan(u, p)
        return solve(NonlinearProblem(nf, u, p), alg; maxiters = 50, abstol = 1.0f-5)
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

# Store-inverse Broyden inverts the initial Jacobian on the first compiled step.
# true_jacobian cases supply an analytic `jac` because AD array Jacobians under
# compile throw (JuliaDiff/DifferentiationInterface.jl#1067). Diagonal Broyden on
# these problems diverges to NaN: host AbsNorm stays MaxIters while compile's
# finite-residual guard returns Unstable, so diagonal cases assert at maxiters=1
# (first-step match) only.
@testset "Broyden first-step inverse under compile: $pname $aname" for (
        pname, f, jac, u0, aname, alg,
    ) in (
        (
            :cube, (u, p) -> u .^ 3 .- p, (u, p) -> diagm(0 => 3 .* u .^ 2),
            Float32[3, -2], :dense, Broyden(),
        ),
        (
            :cube, (u, p) -> u .^ 3 .- p, (u, p) -> diagm(0 => 3 .* u .^ 2),
            Float32[3, -2], :diagonal, Broyden(; update_rule = Val(:diagonal)),
        ),
        (
            :cube, (u, p) -> u .^ 3 .- p, (u, p) -> diagm(0 => 3 .* u .^ 2),
            Float32[3, -2], :true_jacobian,
            Broyden(; init_jacobian = Val(:true_jacobian)),
        ),
        (
            :exp, (u, p) -> exp.(u) .- p, (u, p) -> diagm(0 => exp.(u)),
            Float32[3, 4], :dense, Broyden(),
        ),
        (
            :exp, (u, p) -> exp.(u) .- p, (u, p) -> diagm(0 => exp.(u)),
            Float32[3, 4], :diagonal, Broyden(; update_rule = Val(:diagonal)),
        ),
        (
            :exp, (u, p) -> exp.(u) .- p, (u, p) -> diagm(0 => exp.(u)),
            Float32[3, 4], :true_jacobian,
            Broyden(; init_jacobian = Val(:true_jacobian)),
        ),
    )
    nf = aname === :true_jacobian ? NonlinearFunction(f; jac) : NonlinearFunction(f)
    maxiters = aname === :diagonal ? 1 : 50
    function dosolve_broyden(u, p)
        return solve(
            NonlinearProblem(nf, u, p), alg;
            maxiters = maxiters, abstol = 1.0f-5,
            termination_condition = AbsNormTerminationMode(Base.Fix1(maximum, abs)),
        )
    end
    p_host = Float32[2]
    sol_host = dosolve_broyden(u0, p_host)
    sol = Reactant.@jit dosolve_broyden(Reactant.to_rarray(u0), Reactant.to_rarray(p_host))
    @test sol.retcode isa Reactant.ConcreteEnum{ReturnCode.T}
    @test sol.retcode == sol_host.retcode
    @test Array(sol.u) ≈ Array(sol_host.u) rtol = 1.0f-3
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
