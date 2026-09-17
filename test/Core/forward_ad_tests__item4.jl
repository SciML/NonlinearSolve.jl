using NonlinearSolve, SciMLBase, SciMLStructures
using ForwardDiff, LinearAlgebra
using Test

# ForwardDiff Duals nested inside a structured `p` (NamedTuple, Tuple, or a
# SciMLStructures parameter container) — or Duals present only in `u0` — bypassed
# the `DualAbstractNonlinearProblem` dispatch, which only matches when `p` is
# literally a `Dual` or an `AbstractArray{<:Dual}`. The generic path then ran the
# solver's internal state (e.g. quasi-Newton rank-one updates) in Dual
# arithmetic: update denominators vanish in value while their partials stay
# O(1), so partials diverge while value-only convergence checks report success.
# SciML/NonlinearSolve.jl#1283, SciML/ModelingToolkit.jl#5125.

const NCHAIN = 4

# u_i^3 + u_i + u_{i-1} = p_i — the coupled chain that produced ~1e74 partials
# on the unprotected path.
function chain_f!(du, u, p)
    du[1] = u[1]^3 + u[1] - p[1]
    for i in 2:NCHAIN
        du[i] = u[i]^3 + u[i] + u[i - 1] - p[i]
    end
    return du
end

pvals = [1.5, 2.5, 3.5, 4.5]

function jacobian_chain(u)
    J = zeros(NCHAIN, NCHAIN)
    for i in 1:NCHAIN
        J[i, i] = 3u[i]^2 + 1
        i > 1 && (J[i, i - 1] = 1.0)
    end
    return J
end

# ∂f/∂p = -I ⇒ du/dp = J_u⁻¹
function expected_partials(pv)
    u = solve(
        NonlinearProblem(NonlinearFunction(chain_f!), ones(NCHAIN), pv),
        NewtonRaphson()
    ).u
    return inv(jacobian_chain(u))
end

const EXPECTED = expected_partials(pvals)
const PARTIAL_TOL = 1.0e-6

# Seed direction e_i: dual_j = p_j + δ_ij ε, so partials(u_k)[1] = ∂u_k/∂p_i.
duals_i(i) = ForwardDiff.Dual{:testtag}.(pvals, [j == i ? 1.0 : 0.0 for j in 1:NCHAIN])
extract(sol) = map(uᵢ -> ForwardDiff.partials(uᵢ)[1], sol.u)

@testset "Duals in NamedTuple/Tuple parameters" begin
    for (name, p_of, getp) in (
            ("NamedTuple", θ -> (; tunable = θ), p -> p.tunable),
            ("Tuple", θ -> (θ,), p -> p[1]),
        )
        @testset "$name" for alg in (Broyden(), Klement(), NewtonRaphson())
            prob = NonlinearProblem(
                NonlinearFunction((du, u, p) -> chain_f!(du, u, getp(p))),
                ones(NCHAIN), p_of(duals_i(2))
            )
            sol = solve(prob, alg)
            @test sol.retcode == ReturnCode.Success
            @test ForwardDiff.value.(sol.u) ≈
                solve(
                NonlinearProblem(NonlinearFunction(chain_f!), ones(NCHAIN), pvals), alg,
            ).u
            @test extract(sol) ≈ EXPECTED[:, 2] rtol = PARTIAL_TOL
        end
    end
end

# Minimal SciMLStructures parameter container, mirroring how ModelingToolkit's
# `MTKParameters` exposes its tunable buffer.
struct StructuredParams{T, C}
    tunable::T
    constants::C
end
SciMLStructures.isscimlstructure(::StructuredParams) = true
SciMLStructures.ismutablescimlstructure(::StructuredParams) = true
SciMLStructures.hasportion(::SciMLStructures.Tunable, ::StructuredParams) = true
SciMLStructures.hasportion(::SciMLStructures.Constants, ::StructuredParams) = true
SciMLStructures.hasportion(::SciMLStructures.AbstractPortion, ::StructuredParams) = false
function SciMLStructures.canonicalize(::SciMLStructures.Tunable, p::StructuredParams)
    return p.tunable, vals -> StructuredParams(vals, p.constants), true
end
function SciMLStructures.canonicalize(::SciMLStructures.Constants, p::StructuredParams)
    return p.constants, vals -> StructuredParams(p.tunable, vals), true
end
SciMLStructures.replace(::SciMLStructures.Tunable, p::StructuredParams, vals) =
    StructuredParams(vals, p.constants)
SciMLStructures.replace(::SciMLStructures.Constants, p::StructuredParams, vals) =
    StructuredParams(p.tunable, vals)

@testset "Duals in SciMLStructure parameters" begin
    for (name, p_of) in (
            ("bare SciMLStructure", θ -> StructuredParams(θ, [0.0])),
            (
                "DespecializedParameters",
                θ -> SciMLBase.DespecializedParameters(StructuredParams(θ, [0.0])),
            ),
        )
        @testset "$name" for alg in (Broyden(), Klement(), NewtonRaphson())
            prob = NonlinearProblem(
                NonlinearFunction((du, u, p) -> chain_f!(du, u, p.tunable)),
                ones(NCHAIN), p_of(duals_i(3))
            )
            sol = solve(prob, alg)
            @test sol.retcode == ReturnCode.Success
            @test extract(sol) ≈ EXPECTED[:, 3] rtol = PARTIAL_TOL
        end
    end
end

@testset "Quasi-Newton partial corruption" begin
    # Rosenbrock valley: the root (1, 1) is independent of p[1] (it only scales
    # the first residual), so all partials are structurally zero. The raw-Dual
    # solve reports Success but leaves O(1e-5) partial error — update
    # denominators vanish in value while their partials do not.
    function rosen!(du, u, p)
        du[1] = p[1] * (1 - u[1])
        du[2] = 10 * (u[2] - u[1]^2)
        return du
    end
    for (name, p) in (
            ("NamedTuple", (; tunable = [ForwardDiff.Dual{:testtag}(1.0, 1.0)])),
            ("Tuple", ([ForwardDiff.Dual{:testtag}(1.0, 1.0)],)),
        )
        pfun = p isa NamedTuple ? (p -> p.tunable) : (p -> p[1])
        @testset "$name" for alg in (Broyden(), Klement())
            prob = NonlinearProblem(
                NonlinearFunction((du, u, p) -> rosen!(du, u, pfun(p))),
                [-1.2, 1.0], p
            )
            sol = solve(prob, alg; maxiters = 2000)
            @test sol.retcode == ReturnCode.Success
            @test ForwardDiff.value.(sol.u) ≈ [1.0, 1.0] atol = 1.0e-8
            for uᵢ in sol.u
                @test all(<(1.0e-10) ∘ abs, ForwardDiff.partials(uᵢ))
            end
        end
    end
end

@testset "Duals only in u0" begin
    # A converged root does not depend on the initial guess: partials must be
    # exactly zero under the incoming Dual tag, not accumulated solver history.
    for alg in (Broyden(), NewtonRaphson())
        prob = NonlinearProblem(
            NonlinearFunction(chain_f!),
            ForwardDiff.Dual{:testtag}.(ones(NCHAIN), 1.0),
            pvals
        )
        sol = solve(prob, alg)
        @test sol.retcode == ReturnCode.Success
        @test ForwardDiff.value.(sol.u) ≈
            solve(
            NonlinearProblem(NonlinearFunction(chain_f!), ones(NCHAIN), pvals), alg,
        ).u
        for uᵢ in sol.u
            @test all(iszero, ForwardDiff.partials(uᵢ))
        end
    end
end

@testset "Flat AbstractArray{Dual} p regression" begin
    for alg in (Broyden(), NewtonRaphson())
        prob = NonlinearProblem(
            NonlinearFunction(chain_f!), ones(NCHAIN), duals_i(4)
        )
        sol = solve(prob, alg)
        @test sol.retcode == ReturnCode.Success
        @test extract(sol) ≈ EXPECTED[:, 4] rtol = PARTIAL_TOL
    end
end

@testset "init/solve!/reinit! with structured p" begin
    p_of(θ) = (; tunable = θ)
    prob = NonlinearProblem(
        NonlinearFunction((du, u, p) -> chain_f!(du, u, p.tunable)),
        ones(NCHAIN), p_of(duals_i(1))
    )
    cache = SciMLBase.init(prob, Broyden())
    sol = SciMLBase.solve!(cache)
    @test sol.retcode == ReturnCode.Success
    @test extract(sol) ≈ EXPECTED[:, 1] rtol = PARTIAL_TOL

    p2 = pvals .+ 1.0
    E2 = expected_partials(p2)
    SciMLBase.reinit!(
        cache;
        p = p_of(duals_i(1) .+ 1.0),
        u0 = ones(NCHAIN),
    )
    sol = SciMLBase.solve!(cache)
    @test sol.retcode == ReturnCode.Success
    @test extract(sol) ≈ E2[:, 1] rtol = PARTIAL_TOL
end

@testset "polyalgorithm and default solve with structured p" begin
    for alg in (FastShortcutNonlinearPolyalg(), nothing)
        prob = NonlinearProblem(
            NonlinearFunction((du, u, p) -> chain_f!(du, u, p.tunable)),
            ones(NCHAIN), (; tunable = duals_i(2))
        )
        sol = solve(prob, alg)
        @test sol.retcode == ReturnCode.Success
        @test extract(sol) ≈ EXPECTED[:, 2] rtol = PARTIAL_TOL
    end
end

@testset "IntervalNonlinearProblem with structured p" begin
    # u^3 + u = p: du/dp = 1/(3u²+1) ≈ 0.18460 at p = 3
    p = (; tunable = [ForwardDiff.Dual{:testtag}(3.0, 1.0)])
    prob = IntervalNonlinearProblem(
        (u, p) -> u^3 + u - p.tunable[1], (0.0, 4.0), p
    )
    # `solve(prob)` goes through the NonlinearSolveBase funnel; explicit
    # bracketing algorithms dispatch through `bracketingnonlinear_solve_up`.
    for alg in (nothing, Bisection(), ModAB())
        sol = alg === nothing ? solve(prob) : solve(prob, alg; abstol = 1.0e-10)
        # FloatingPointLimit is a converged bracketing termination: the interval
        # endpoints become adjacent Float64s before abstol is met.
        @test sol.retcode in (ReturnCode.Success, ReturnCode.FloatingPointLimit)
        @test ForwardDiff.value(sol.u) ≈ 1.2134116627622296 rtol = 1.0e-6
        @test ForwardDiff.partials(sol.u)[1] ≈ 0.18460049422892552 rtol = PARTIAL_TOL
    end
end

@testset "Simple solvers with structured p" begin
    # SimpleNonlinearSolve dispatches through `simplenonlinearsolve_solve_up`,
    # bypassing the NonlinearSolveBase `solve_call`/`init_call` funnels.
    f(u, p) = u^3 + u - p.tunable[1]
    for alg in (SimpleNewtonRaphson(), SimpleBroyden())
        prob = NonlinearProblem{false}(
            f, 1.0, (; tunable = [ForwardDiff.Dual{:testtag}(3.0, 1.0)])
        )
        sol = solve(prob, alg)
        @test sol.retcode == ReturnCode.Success
        @test ForwardDiff.value(sol.u) ≈ 1.2134116627622296 rtol = 1.0e-8
        @test ForwardDiff.partials(sol.u)[1] ≈ 0.18460049422892552 rtol = PARTIAL_TOL
    end
end

@testset "Dict parameters" begin
    # Concretely-typed `Dict{Symbol, Dual}` cannot hold stripped primal values
    # (`convert` re-wraps them), so the view must fall back to a plain `Dict`
    # rather than recurse forever.
    d = ForwardDiff.Dual{:testtag}(3.0, 1.0)
    for (name, p) in (
            ("Dict{Symbol, Any}", Dict{Symbol, Any}(:a => d)),
            ("Dict{Symbol, Dual}", Dict{Symbol, typeof(d)}(:a => d)),
        )
        @testset "$name" for alg in (Broyden(), NewtonRaphson())
            prob = NonlinearProblem{false}(
                (u, p) -> u^3 + u - p[:a], 1.0, p
            )
            sol = solve(prob, alg)
            @test sol.retcode == ReturnCode.Success
            @test ForwardDiff.value(sol.u) ≈ 1.2134116627622296 rtol = 1.0e-8
            @test ForwardDiff.partials(sol.u)[1] ≈ 0.18460049422892552 rtol = PARTIAL_TOL
        end
    end
end

@testset "NonlinearLeastSquaresProblem with structured p" begin
    p = (; tunable = [ForwardDiff.Dual{:testtag}(3.0, 1.0)])
    fnlls(u, p) = [u[1] - p.tunable[1], u[2] - 2p.tunable[1]]
    prob = NonlinearLeastSquaresProblem(
        NonlinearFunction(fnlls, resid_prototype = zeros(2)), [1.0, 1.0], p
    )
    sol = solve(prob, GaussNewton())
    @test sol.retcode == ReturnCode.Success
    @test ForwardDiff.value.(sol.u) ≈ [3.0, 6.0] atol = 1.0e-8
    @test ForwardDiff.partials(sol.u[1])[1] ≈ 1.0 atol = 1.0e-8
    @test ForwardDiff.partials(sol.u[2])[1] ≈ 2.0 atol = 1.0e-8
end
