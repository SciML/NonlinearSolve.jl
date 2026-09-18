using NonlinearSolveFirstOrder, NonlinearSolveBase, LinearAlgebra, StableRNGs
using SparseArrays, StaticArrays, LinearSolve

const NB = NonlinearSolveBase
const IAPI = NB.InternalAPI

# Moré subproblem on dense underdetermined (`m < n`) Jacobians. For `2m ≤ n` the
# `lmpar` workspace switches to the Gram formulation `B = J D⁻¹`, `G = BBᵀ`,
# `(G + λI) w = -fu`, `p = D⁻¹ Bᵀ w`: per-λ work is an `m × m` Cholesky instead of
# an `n²` Givens sweep, and `λ = 0` becomes a legitimate minimum-norm
# Gauss-Newton step (the padded `n × n` QR is always rank-deficient and can only
# approach it through `λ → 0`).

function more_descent_cache(f, u0, J0, fu0; kwargs...)
    prob = NonlinearLeastSquaresProblem(NonlinearFunction{false}(f), u0)
    stats = SciMLBase.NLStats(0, 0, 0, 0, 0)
    return IAPI.init(prob, NB.MoreTrustRegionDescent(; kwargs...), J0, fu0, u0; stats)
end

@testset "More subproblem: Gram mode on wide Jacobians" begin
    rng = StableRNG(0)
    m, n = 3, 10
    A = randn(rng, m, n)
    b = randn(rng, m)
    f_lin(u, p) = A * u - b
    fu0 = f_lin(zeros(n), nothing)

    # `2m ≤ n` selects the Gram workspace; the same concrete `lincache` type is
    # kept either way so `init` stays inferred
    cache = more_descent_cache(f_lin, zeros(n), A, fu0)
    @test cache.lincache isa NB._MoreLmparWorkspace && cache.lincache.gram
    @inferred IAPI.init(
        NonlinearLeastSquaresProblem(NonlinearFunction{false}(f_lin), zeros(n)),
        NB.MoreTrustRegionDescent(), A, fu0, zeros(n);
        stats = SciMLBase.NLStats(0, 0, 0, 0, 0)
    )

    # λ = 0 is reachable: a full-rank wide `J` has a nonsingular `G = JJᵀ`, so an
    # interior step is the minimum-norm Gauss-Newton step exactly — the padded
    # `lmpar` mode can only approach it (it always sees `nsing < n`)
    res = IAPI.solve!(cache, A, fu0, zeros(n); trust_region = 1.0e6)
    @test res.success
    @test iszero(res.extras.λ)
    @test res.δu ≈ A \ b atol = 1.0e-10
    @test res.extras.step_norm ≈ norm(res.δu)

    # `n/2 < m < n` keeps the padded `lmpar` mode (its O(n²) sweep beats the
    # m³ Cholesky there); square/overdetermined Jacobians do too
    for (mm, nn) in ((3, 4), (4, 4), (6, 4))
        Am = randn(rng, mm, nn)
        bm = randn(rng, mm)
        fc = more_descent_cache((u, p) -> Am * u - bm, zeros(nn), Am, -bm)
        @test fc.lincache isa NB._MoreLmparWorkspace && !fc.lincache.gram
    end

    # Damped trials: at the returned `λ > 0` the step equals the exact solution
    # of `(JᵀJ + λI) p = -Jᵀf` and sits on the trust-region boundary. All `Δ`s
    # stay below the minimum-norm step length so the subproblem is constrained
    pmin_norm = norm(A \ b)
    for Δ in (0.05, 0.3, 0.6) .* pmin_norm
        cache.λ = 0.0
        cache.gn_valid = false
        res = IAPI.solve!(cache, A, fu0, zeros(n); trust_region = Δ)
        @test res.success
        λ = res.extras.λ
        @test λ > 0
        p_ref = (A' * A + λ * I) \ (A' * b)
        @test res.δu ≈ p_ref rtol = 1.0e-8
        @test abs(norm(p_ref) - Δ) <= cache.θ * Δ
    end

    # The same subproblem solved by the padded `lmpar` path: Gram and padded
    # iterate on the same φ(λ) = ‖Dp‖ - Δ and must land on the same step
    padded = more_descent_cache(f_lin, zeros(n), A, fu0)
    padded.lincache = NB._more_lmpar_workspace(A, padded.lincache.stats)
    for Δ in (0.05, 0.3, 0.6) .* pmin_norm
        padded.λ = 0.0
        padded.gn_valid = false
        res_p = IAPI.solve!(padded, A, fu0, zeros(n); trust_region = Δ)
        cache.λ = 0.0
        cache.gn_valid = false
        res_g = IAPI.solve!(cache, A, fu0, zeros(n); trust_region = Δ)
        @test res_p.success && res_g.success
        @test res_g.δu ≈ res_p.δu atol = 1.0e-8
    end
end

@testset "More subproblem: Gram mode with Jacobian scaling" begin
    rng = StableRNG(1)
    m, n = 3, 12
    A = randn(rng, m, n)
    A[:, 1] .*= 1.0e4 # badly scaled column so `D` differs meaningfully from `I`
    b = randn(rng, m)
    f_lin(u, p) = A * u - b
    fu0 = f_lin(zeros(n), nothing)

    cache = more_descent_cache(
        f_lin, zeros(n), A, fu0; scaling = TrustRegionScaling.Jacobian
    )
    @test cache.lincache.gram
    dtd = cache.dtd
    @test dtd isa Vector{Float64}

    # Undamped step is the minimum-‖Dp‖ solution `D⁻² Jᵀ (J D⁻² Jᵀ)⁻¹ b`
    res = IAPI.solve!(cache, A, fu0, zeros(n); trust_region = 1.0e6)
    @test res.success
    @test iszero(res.extras.λ)
    D2 = Diagonal(dtd)
    @test res.δu ≈ inv(D2) * A' * ((A * inv(D2) * A') \ b) rtol = 1.0e-8

    # Damped: `p` solves `(JᵀJ + λD²) p = -Jᵀf` and `‖Dp‖ = Δ` at `λ > 0`
    pmin_Dnorm = norm(sqrt.(dtd) .* res.δu)
    for Δ in (0.1, 0.5) .* pmin_Dnorm
        cache.λ = 0.0
        cache.gn_valid = false
        res = IAPI.solve!(cache, A, fu0, zeros(n); trust_region = Δ)
        @test res.success
        λ = res.extras.λ
        @test λ > 0
        p_ref = (A' * A + λ * D2) \ (A' * b)
        @test res.δu ≈ p_ref rtol = 1.0e-8
        @test abs(norm(sqrt.(dtd) .* p_ref) - Δ) <= cache.θ * Δ
    end
end

@testset "More subproblem: underdetermined end-to-end solves" begin
    rng = StableRNG(2)

    # tp293-class: a single nonlinear residual in 50 unknowns. The trajectory of
    # minimum-norm steps from `u0` stays in `span(u0)`, converging to `u0/‖u0‖`
    f_scalar(u, p) = [sum(abs2, u) - 1.0]
    u0 = fill(0.3, 50)
    prob = NonlinearLeastSquaresProblem(NonlinearFunction{false}(f_scalar), u0)
    alg = TrustRegion(; subproblem = TrustRegionSubproblem.More)
    sol = solve(prob, alg; maxiters = 100, abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol)
    @test norm(sol.resid) < 1.0e-8
    @test sol.u ≈ u0 ./ norm(u0) rtol = 1.0e-6
    @test 0 < sol.stats.nsteps < 100
    @test sol.stats.nfactors > 0
    @test sol.stats.nsolve > 0

    # the subproblem cache actually ran in Gram mode
    scache = init(prob, alg; abstol = 1.0e-10)
    @test scache.descent_cache.lincache.gram

    # linear `m = 3, n = 10` wide system: minimum-norm solution from `u0 = 0`
    A = randn(rng, 3, 10)
    b = A * randn(rng, 10)
    prob_lin = NonlinearLeastSquaresProblem(
        NonlinearFunction{false}((u, p) -> A * u - b), zeros(10)
    )
    sol_lin = solve(prob_lin, alg; maxiters = 100, abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol_lin)
    @test norm(sol_lin.resid) < 1.0e-8
    @test sol_lin.u ≈ A \ b atol = 1.0e-8

    # rank-deficient wide Jacobian (duplicated row, consistent rhs): `G` is
    # singular so `λ = 0` is declined and the damped iteration recovers the
    # minimum-norm solution
    A_rd = A[[1, 1, 2], :]
    b_rd = A_rd * randn(rng, 10)
    prob_rd = NonlinearLeastSquaresProblem(
        NonlinearFunction{false}((u, p) -> A_rd * u - b_rd), zeros(10)
    )
    sol_rd = solve(prob_rd, alg; maxiters = 100, abstol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol_rd)
    @test norm(sol_rd.resid) < 1.0e-8
    @test sol_rd.u ≈ pinv(A_rd) * b_rd atol = 1.0e-6

    # Float32 wide problem
    A32 = Float32.(A)
    b32 = Float32.(b)
    prob32 = NonlinearLeastSquaresProblem(
        NonlinearFunction{false}((u, p) -> A32 * u - b32), zeros(Float32, 10)
    )
    sol32 = solve(prob32, alg; maxiters = 100, abstol = 1.0f-4)
    @test SciMLBase.successful_retcode(sol32)
    @test norm(sol32.resid) < 1.0f-3
    @test sol32.u ≈ Float32.(A \ b) atol = 1.0f-3
end

@testset "More subproblem: Gram mode leaves other paths untouched" begin
    rng = StableRNG(3)
    m, n = 3, 10
    A = randn(rng, m, n)
    b = randn(rng, m)
    f_lin(u, p) = A * u - b
    alg = TrustRegion(; subproblem = TrustRegionSubproblem.More)

    # an explicit `linsolve` is honored — the damped system goes through
    # LinearSolve rather than the internal Gram factorization
    prob_lu = NonlinearLeastSquaresProblem(NonlinearFunction{false}(f_lin), zeros(n))
    cache_lu = init(
        prob_lu,
        TrustRegion(;
            subproblem = MoreTrustRegionDescent(; linsolve = LUFactorization())
        )
    )
    @test !(cache_lu.descent_cache.lincache isa NB._MoreLmparWorkspace)
    sol_lu = solve(
        prob_lu,
        TrustRegion(;
            subproblem = MoreTrustRegionDescent(; linsolve = LUFactorization())
        );
        maxiters = 100, abstol = 1.0e-8
    )
    @test SciMLBase.successful_retcode(sol_lu)
    @test norm(sol_lu.resid) < 1.0e-6

    # sparse `jac_prototype` stays on the generic augmented-system path
    f_lin!(F, u, p) = (F .= A * u - b)
    fn_sp = NonlinearFunction{true}(
        f_lin!; resid_prototype = zeros(m),
        jac_prototype = sparse(A)
    )
    prob_sp = NonlinearLeastSquaresProblem(fn_sp, zeros(n))
    cache_sp = init(prob_sp, alg)
    @test !(cache_sp.descent_cache.lincache isa NB._MoreLmparWorkspace)
    sol_sp = solve(prob_sp, alg; maxiters = 100, abstol = 1.0e-8)
    @test SciMLBase.successful_retcode(sol_sp)
    @test norm(sol_sp.resid) < 1.0e-6

    # StaticArray problems keep the normal-form (non-`lmpar`) workspace
    As = SMatrix{m, n}(A)
    bs = SVector{m}(b)
    f_sa(u, p) = As * u - bs
    prob_sa = NonlinearLeastSquaresProblem(
        NonlinearFunction{false}(f_sa), @SVector(zeros(n))
    )
    cache_sa = init(prob_sa, alg)
    @test !(cache_sa.descent_cache.lincache isa NB._MoreLmparWorkspace)
    sol_sa = solve(prob_sa, alg; maxiters = 1000, abstol = 1.0e-6)
    @test SciMLBase.successful_retcode(sol_sa)
    @test norm(sol_sa.resid) < 1.0e-5
end
