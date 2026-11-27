using NonlinearSolve

using CUDA, NonlinearSolve, LinearSolve, StableRNGs, ADTypes
using LinearAlgebra: norm

# Empty stderr, or exactly one deprecation warning for `GPUCompiler.compile_hook`, optionally
# with the `likely near <file>:<line>` footer that Julia prints after it
const COMPILE_HOOK_DEPWARN = r"\AWARNING: Use of GPUCompiler\.compile_hook is deprecated[^\n]*\n(?:  likely near [^\n]+:\d+\n)?\z"
only_compile_hook_warning(s) = isempty(s) || occursin(COMPILE_HOOK_DEPWARN, s)

@testset "compile_hook warning matcher" begin
    w = "WARNING: Use of GPUCompiler.compile_hook is deprecated, use X instead.\n  likely near C:\\a\\b.jl:12\n"
    @test only_compile_hook_warning(w)
    @test only_compile_hook_warning("")
    @test !only_compile_hook_warning(w * w)
    @test only_compile_hook_warning(first(w, findfirst('\n', w))) # no footer
    @test !only_compile_hook_warning(w * "  unrelated indented diagnostic\n")
    # A footer must be a source location ending in `:<line>`
    dep = "WARNING: Use of GPUCompiler.compile_hook is deprecated, use X instead.\n"
    @test !only_compile_hook_warning(dep * "  likely near an unrelated diagnostic\n")
    @test !only_compile_hook_warning(dep * "  likely near C:\\a\\b.jl:\n")
    @test !only_compile_hook_warning("┌ Warning: something else\n" * w)
end

if CUDA.functional()
    CUDA.allowscalar(false)

    A = cu(rand(StableRNG(0), 4, 4))
    u0 = cu(rand(StableRNG(0), 4))
    b = cu(rand(StableRNG(0), 4))

    linear_f(du, u, p) = (du .= A * u .+ b)

    prob = NonlinearProblem(linear_f, u0)
    iip_prob = prob

    # ForwardDiff uses scalar indexing which doesn't work on GPU
    # Use AutoFiniteDiff for GPU-compatible Jacobian computation
    fd_autodiff = AutoFiniteDiff()

    SOLVERS = (
        NewtonRaphson(; autodiff = fd_autodiff),
        LevenbergMarquardt(; linsolve = QRFactorization(), autodiff = fd_autodiff),
        LevenbergMarquardt(; linsolve = KrylovJL_GMRES(), autodiff = fd_autodiff),
        PseudoTransient(; autodiff = fd_autodiff),
        Klement(),
        Broyden(; linesearch = LiFukushimaLineSearch()),
        LimitedMemoryBroyden(; threshold = 2, linesearch = LiFukushimaLineSearch()),
        DFSane(),
        TrustRegion(; linsolve = QRFactorization(), autodiff = fd_autodiff),
        TrustRegion(; linsolve = KrylovJL_GMRES(), concrete_jac = true, autodiff = fd_autodiff),  # Needed if Zygote not loaded
    )

    # LinearSolve caps KernelAbstractions below 0.10, which holds CUDA below 6.5. CUDA 6.4
    # reads the deprecated binding `GPUCompiler.compile_hook` and the resulting warning is
    # printed only once, in the first solve. That solve may print this one deprecation (a
    # `WARNING: Use of ...` line and its `likely near <file>:<line>` footer) and nothing
    # else; every other solve must print nothing. Use `@test_nowarn` for the first solve
    # too once CUDA 6.5 resolves in this environment.
    @testset "[IIP] GPU Solvers" begin
        @testset "$(nameof(typeof(alg)))" for (i, alg) in enumerate(SOLVERS)
            if i == 1
                @test_warn only_compile_hook_warning solve(
                    prob, alg; abstol = 1.0f-5, reltol = 1.0f-5
                )
            else
                @test_nowarn sol = solve(prob, alg; abstol = 1.0f-5, reltol = 1.0f-5)
            end
        end
    end

    linear_f(u, p) = A * u .+ b

    prob = NonlinearProblem{false}(linear_f, u0)
    oop_prob = prob

    @testset "[OOP] GPU Solvers" begin
        @testset "$(nameof(typeof(alg)))" for alg in SOLVERS
            @test_nowarn sol = solve(prob, alg; abstol = 1.0f-5, reltol = 1.0f-5)
        end
    end

    @testset "Broyden inverse initialization" begin
        for prob in (iip_prob, oop_prob)
            sol = solve(
                prob, Broyden(; linesearch = LiFukushimaLineSearch());
                abstol = 1.0f-5, reltol = 1.0f-5
            )
            @test norm(A * sol.u + b) < 1.0f-4
        end
    end
end
