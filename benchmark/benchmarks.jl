using NonlinearSolve, BenchmarkTools
using LinearAlgebra, StableRNGs

const SUITE = BenchmarkGroup()

# =============================================================================
# Small Dense Problems
# =============================================================================

SUITE["small_dense"] = BenchmarkGroup()

# Powell singular function (standard 4D test problem)
function powell!(du, u, p)
    du[1] = u[1] + 10u[2]
    du[2] = sqrt(5) * (u[3] - u[4])
    du[3] = (u[2] - 2u[3])^2
    du[4] = sqrt(10) * (u[1] - u[4])^2
    return nothing
end
powell_prob = NonlinearProblem(powell!, [3.0, -1.0, 0.0, 1.0])

# Rosenbrock (2D)
function rosenbrock!(du, u, p)
    du[1] = 1 - u[1]
    du[2] = 10 * (u[2] - u[1]^2)
    return nothing
end
rosenbrock_prob = NonlinearProblem(rosenbrock!, [-1.2, 1.0])

SUITE["small_dense"]["powell"] = BenchmarkGroup()
SUITE["small_dense"]["powell"]["NewtonRaphson"] = @benchmarkable solve(
    $powell_prob, NewtonRaphson()
)
SUITE["small_dense"]["powell"]["TrustRegion"] = @benchmarkable solve(
    $powell_prob, TrustRegion()
)
SUITE["small_dense"]["rosenbrock"] = BenchmarkGroup()
SUITE["small_dense"]["rosenbrock"]["Polyalg"] = @benchmarkable solve(
    $rosenbrock_prob, FastShortcutNonlinearPolyalg()
)

# =============================================================================
# Medium Problem: steady-state nonlinear diffusion (in-place)
# =============================================================================

SUITE["medium"] = BenchmarkGroup()

# -u'' + exp(u) = 0 on (0,1), u(0)=1, u(1)=0
function nonlinear_poisson!(F, u, p)
    N, invdx2 = p
    F[1] = u[1] - 1.0
    F[N] = u[N]
    for i in 2:(N - 1)
        F[i] = (u[i - 1] - 2u[i] + u[i + 1]) * invdx2 + exp(u[i])
    end
    return nothing
end

const NLP_N = 200
nlp_prob = NonlinearProblem(
    nonlinear_poisson!, 0.5 * ones(NLP_N), (NLP_N, (NLP_N - 1)^2)
)

SUITE["medium"]["nonlinear_poisson"] = BenchmarkGroup()
SUITE["medium"]["nonlinear_poisson"]["NewtonRaphson"] = @benchmarkable solve(
    $nlp_prob, NewtonRaphson()
)
SUITE["medium"]["nonlinear_poisson"]["PseudoTransient"] = @benchmarkable solve(
    $nlp_prob, PseudoTransient()
)
SUITE["medium"]["nonlinear_poisson"]["LimitedMemoryBroyden"] = @benchmarkable solve(
    $nlp_prob, LimitedMemoryBroyden()
)
SUITE["medium"]["nonlinear_poisson"]["DFSane"] = @benchmarkable solve(
    $nlp_prob, DFSane()
)

# =============================================================================
# Nonlinear Least Squares
# =============================================================================

SUITE["least_squares"] = BenchmarkGroup()

# Fit y = a * exp(b * x) to noisy data
const NLS_NDATA = 100
function nls_residual!(r, v, p)
    x, y = p
    for i in eachindex(x)
        r[i] = v[1] * exp(v[2] * x[i]) - y[i]
    end
    return nothing
end

const nls_rng = StableRNG(123)
const nls_x = collect(range(0.0, 2.0, length = NLS_NDATA))
const nls_y = 2.0 .* exp.(0.5 .* nls_x) .+ 0.05 .* randn(nls_rng, NLS_NDATA)
nls_f = NonlinearFunction(nls_residual!; resid_prototype = zeros(NLS_NDATA))
nls_prob = NonlinearLeastSquaresProblem(nls_f, [1.0, 0.0], (nls_x, nls_y))

SUITE["least_squares"]["exp_fit"] = BenchmarkGroup()
SUITE["least_squares"]["exp_fit"]["LevenbergMarquardt"] = @benchmarkable solve(
    $nls_prob, LevenbergMarquardt()
)
SUITE["least_squares"]["exp_fit"]["GaussNewton"] = @benchmarkable solve(
    $nls_prob, GaussNewton()
)

# =============================================================================
# Simple Solvers (allocation-light path)
# =============================================================================

SUITE["simple"] = BenchmarkGroup()

scalar_prob = NonlinearProblem{false}((u, p) -> u^2 - p, 1.0, 2.0)
scalar_prob_bracket = IntervalNonlinearProblem{false}((u, p) -> u^2 - p, (0.0, 2.0), 2.0)
small_prob = NonlinearProblem{false}((u, p) -> u .^ 2 .- p, ones(4), 2 * ones(4))

SUITE["simple"]["scalar"] = BenchmarkGroup()
SUITE["simple"]["scalar"]["SimpleNewtonRaphson"] = @benchmarkable solve(
    $scalar_prob, SimpleNewtonRaphson()
)
SUITE["simple"]["scalar"]["ITP"] = @benchmarkable solve(
    $scalar_prob_bracket, ITP()
)
SUITE["simple"]["small_system"] = BenchmarkGroup()
SUITE["simple"]["small_system"]["SimpleDFSane"] = @benchmarkable solve(
    $small_prob, SimpleDFSane()
)
