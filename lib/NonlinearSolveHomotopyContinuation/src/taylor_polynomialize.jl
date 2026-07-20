"""
    TaylorHomotopyContinuationJL{AllRoots}(; degree = 3, autodiff = true, kwargs...)
    TaylorHomotopyContinuationJL(; kwargs...) = TaylorHomotopyContinuationJL{false}(; kwargs...)

A solver for general (non-polynomial) nonlinear systems built on top of
HomotopyContinuation.jl. The system is approximated by its multivariate Taylor
polynomial of total degree `degree` around the initial guess, all roots of the
polynomial surrogate are found via homotopy continuation, and every (near-)real
root is used as an initial guess for a Newton iteration on the original system.
Newton iterates that converge are reported as roots.

Unlike [`HomotopyContinuationJL`](@ref) this does not require the system to be
polynomial. The system function must accept `TaylorSeries.jl` jet types as state
(the same operator-overloading requirement as ForwardDiff.jl duals). If the system
*is* polynomial of total degree at most `degree`, the surrogate is exact and all
roots of the system are found.

The `AllRoots` type parameter can be `true` or `false`. If `true`, an
`EnsembleSolution` of all distinct roots found is returned and the initial guess
only serves as the expansion point. If `false`, the single converged root closest
to the initial guess is returned.

The number of homotopy paths grows like `degree^n` where `n` is the number of
unknowns, which restricts this method to small-to-medium systems (roughly
`n ≤ 12` at `degree = 2`, `n ≤ 8` at `degree = 3`).

# Keyword arguments

  - `degree`: the total degree of the Taylor approximation. Higher degrees give more
    faithful surrogates (and find more roots of transcendental systems) at the cost of
    `degree^n` path growth.
  - `autodiff`: the autodiff algorithm used for the Newton polish. `true` maps to
    `AutoForwardDiff()`, `false` to `AutoFiniteDiff()`; any ADTypes.jl algorithm is
    accepted. If the `NonlinearFunction` provides a jacobian it is used directly.

All other keyword arguments are forwarded to `HomotopyContinuation.solve`.
"""
@concrete struct TaylorHomotopyContinuationJL{AllRoots} <:
    NonlinearSolveBase.AbstractNonlinearSolveAlgorithm
    degree::Int
    autodiff
    kwargs
end

function TaylorHomotopyContinuationJL{AllRoots}(;
        degree = 3, autodiff = true, kwargs...
    ) where {AllRoots}
    degree >= 1 || throw(ArgumentError("`degree` must be at least 1"))
    if autodiff isa Bool
        autodiff = autodiff ? AutoForwardDiff() : AutoFiniteDiff()
    end
    return TaylorHomotopyContinuationJL{AllRoots}(degree, autodiff, kwargs)
end

TaylorHomotopyContinuationJL(; kwargs...) = TaylorHomotopyContinuationJL{false}(; kwargs...)

function TaylorHomotopyContinuationJL(
        alg::TaylorHomotopyContinuationJL{R}; kwargs...
    ) where {R}
    return TaylorHomotopyContinuationJL{R}(;
        degree = alg.degree, autodiff = alg.autodiff, alg.kwargs..., kwargs...
    )
end

# `TS.variables!` rebuilds the entire jet space, including multiplication
# tables, on every call. Cache the jet variables per (numvars, order); TaylorN
# arithmetic uses the space carried by the operands so reuse is safe.
const TAYLOR_VAR_CACHE = Dict{Tuple{Int, Int}, Vector{TS.TaylorN{Float64}}}()
const TAYLOR_VAR_LOCK = ReentrantLock()

function taylor_jet_variables(n::Int, degree::Int)
    return lock(TAYLOR_VAR_LOCK) do
        get!(TAYLOR_VAR_CACHE, (n, degree)) do
            TS.variables!("Δ", numvars = n, order = degree, nowarn = true)
        end
    end
end

"""
    $(TYPEDSIGNATURES)

Convert a `TaylorSeries.TaylorN` polynomial (in the deviation from the expansion
point) into a `HomotopyContinuation.ModelKit.Expression` in `vars`. Returns the
expression and the actual total degree of the polynomial.
"""
function taylorn_to_expression(poly::TS.TaylorN, vars)
    ex = HC.ModelKit.Expression(0)
    deg = 0
    for (kp1, homog) in enumerate(poly.coeffs)
        exponents = poly.space.coeff_table[kp1]
        for (i, c) in enumerate(homog.coeffs)
            iszero(c) && continue
            deg = max(deg, kp1 - 1)
            mono = HC.ModelKit.Expression(1)
            for (j, e) in enumerate(exponents[i])
                e > 0 && (mono *= vars[j]^e)
            end
            ex += c * mono
        end
    end
    return ex, deg
end

"""
    $(TYPEDSIGNATURES)

Build the polynomial surrogate system for `f` (of the given
`HomotopySystemVariant`) around expansion point `u0` by evaluating `f` on
TaylorSeries.jl jets. Returns `(system, bezout)` where `system` is an
`HC.ModelKit.System` in the deviation variables and `bezout` is the Bezout
number of the truncation (product of actual total degrees).
"""
function taylor_surrogate_system(f::F, variant, u0, p, degree) where {F}
    if variant == Scalar
        t = TS.Taylor1(Float64, degree)
        fT = f(u0 + t, p)
        var = HC.ModelKit.Variable(:Δx)
        coeffs = fT isa TS.Taylor1 ? fT.coeffs : vcat(fT, zeros(degree))
        ex = sum(coeffs[k + 1] * var^k for k in 0:(length(coeffs) - 1))
        actual_degree = something(findlast(!iszero, coeffs), 2) - 1
        sys = HC.ModelKit.System([ex], variables = [var])
        return sys, max(actual_degree, 1)
    end

    n = length(u0)
    d = taylor_jet_variables(n, degree)
    uT = u0 .+ d
    fT = if variant == Inplace
        buffer = [zero(first(d)) for _ in 1:n]
        f(buffer, uT, p)
        buffer
    else
        f(uT, p)
    end
    length(fT) == n || throw(
        ArgumentError(
            "`TaylorHomotopyContinuationJL` only supports fully determined systems; " *
                "got $(length(fT)) equations in $n unknowns."
        )
    )
    vars = HC.variables(:Δx, 1:n)
    exprs = Vector{HC.ModelKit.Expression}(undef, n)
    bezout = 1
    for i in 1:n
        exprs[i], deg_i = taylorn_to_expression(fT[i], vars)
        bezout *= max(deg_i, 1)
    end
    sys = HC.ModelKit.System(exprs, variables = collect(vars))
    return sys, bezout
end

"""
    $(TYPEDSIGNATURES)

Solve the polynomial surrogate with HomotopyContinuation.jl and return all finite
path endpoints as real candidate vectors (in deviation coordinates). Near-real
endpoints come first; real parts of genuinely complex endpoints are appended as
lower-quality candidates. Endpoints are deduplicated.

`compile = false` is used because every call builds a system with fresh numeric
coefficients, so compiled straight-line programs can never be reused; interpreted
tracking avoids paying seconds of compilation per solve. Threading is only enabled
when the number of paths is large enough to amortize task-scheduling overhead.
Dense Taylor truncations gain nothing from polyhedral start systems (the mixed
volume equals the Bezout number), so the cheaper total-degree start system is used
unless the path count is large.
"""
function solve_taylor_surrogate(sys, bezout; real_tol = 1.0e-6, kwargs...)
    start_system = bezout <= 200 ? :total_degree : :polyhedral
    threading = bezout > 4 * Threads.nthreads()
    result = HC.solve(
        sys; show_progress = false, compile = false, threading, start_system, kwargs...
    )
    endpoints = [HC.solution(pr) for pr in HC.path_results(result)]
    filter!(endpoints) do s
        all(z -> isfinite(real(z)) && isfinite(imag(z)), s) && maximum(abs, s) < 1.0e10
    end
    isrealsol(s) = maximum(abs ∘ imag, s; init = 0.0) < real_tol * (1 + maximum(abs, s))
    candidates = [real.(s) for s in endpoints if isrealsol(s)]
    append!(candidates, [real.(s) for s in endpoints if !isrealsol(s)])
    deduped = Vector{Vector{Float64}}()
    for c in candidates
        if all(d -> maximum(abs, d - c) > 1.0e-8 * (1 + maximum(abs, c)), deduped)
            push!(deduped, c)
        end
    end
    return deduped, result
end

# Real-arithmetic residual+jacobian wrappers for the Newton polish, mirroring
# the complex-valued machinery in jacobian_handling.jl.
@concrete struct PolishFunction{variant <: HomotopySystemVariant}
    f
end

function (pf::PolishFunction{OutOfPlace})(x::AbstractVector, p)
    return pf.f(x, p)
end
function (pf::PolishFunction{Inplace})(u::AbstractVector, x::AbstractVector, p)
    pf.f(u, x, p)
    return u
end
function (pf::PolishFunction{Scalar})(u::AbstractVector, x::AbstractVector, p)
    u[1] = pf.f(x[1], p)
    return u
end

function construct_polish_jacobian(pf::PolishFunction{OutOfPlace}, autodiff, u0, p)
    prep = DI.prepare_jacobian(pf, autodiff, u0, DI.Constant(p), strict = Val(false))
    return (x, p) -> DI.value_and_jacobian(pf, prep, autodiff, x, DI.Constant(p))
end
function construct_polish_jacobian(
        pf::PolishFunction{variant}, autodiff, u0, p
    ) where {variant}
    resid = Vector{Float64}(undef, length(u0))
    prep = DI.prepare_jacobian(
        pf, resid, autodiff, copy(u0), DI.Constant(p), strict = Val(false)
    )
    return function (x, p)
        fu, J = DI.value_and_jacobian(pf, resid, prep, autodiff, x, DI.Constant(p))
        return copy(fu), J
    end
end

"""
    $(TYPEDSIGNATURES)

Newton-polish the candidate `c` (a real vector in the coordinates of the original
system) using `vjac = (x, p) -> (f(x), J(x))`. Success is judged by the achieved
residual rather than step-size convergence: at singular roots Newton converges only
linearly and would report failure despite an excellent iterate. Returns
`(u, converged)`.
"""
function newton_polish(vjac::J, c, p, abstol, maxiters) where {J}
    u = copy(c)
    best_u = copy(c)
    best_resid = Inf
    for _ in 1:maxiters
        fu, jac = vjac(u, p)
        resid = maximum(abs, fu)
        if resid < best_resid
            best_resid = resid
            copyto!(best_u, u)
        end
        resid <= abstol && break
        step = try
            LinearAlgebra.qr(jac, LinearAlgebra.ColumnNorm()) \ fu
        catch
            break
        end
        all(isfinite, step) || break
        u .-= step
    end
    return best_u, best_resid <= sqrt(abstol)
end

function taylor_homotopy_preprocessing(
        prob::NonlinearProblem, alg::TaylorHomotopyContinuationJL;
        abstol, maxiters, kwargs...
    )
    f = if prob.f isa HomotopyNonlinearFunction
        prob.f
    else
        HomotopyNonlinearFunction(prob.f)
    end

    u0 = state_values(prob)
    p = parameter_values(prob)
    isscalar = u0 isa Number
    iip = SciMLBase.isinplace(prob)
    variant = iip ? Inplace : isscalar ? Scalar : OutOfPlace

    u0_p = f.polynomialize(u0, p)
    expansion_point = isscalar ? u0_p : convert(Vector{Float64}, u0_p)

    sys, bezout = taylor_surrogate_system(f.f.f, variant, u0_p, p, alg.degree)
    candidates, orig_sol = solve_taylor_surrogate(sys, bezout; alg.kwargs..., kwargs...)

    pf = PolishFunction{variant}(f.f.f)
    polish_u0 = isscalar ? [expansion_point] : expansion_point
    vjac = if SciMLBase.has_jac(f.f)
        ExplicitPolishJacobian{variant}(pf, f.f.jac)
    else
        construct_polish_jacobian(pf, alg.autodiff, polish_u0, p)
    end

    roots = Vector{Vector{Float64}}()
    for c in candidates
        guess = polish_u0 .+ c
        u, converged = newton_polish(vjac, guess, p, abstol, maxiters)
        converged || continue
        if all(r -> maximum(abs, r - u) > 1.0e-6 * (1 + maximum(abs, u)), roots)
            push!(roots, u)
        end
    end

    return f, roots, expansion_point, orig_sol
end

@concrete struct ExplicitPolishJacobian{variant}
    pf
    jac
end

function (f::ExplicitPolishJacobian{OutOfPlace})(x, p)
    return f.pf(x, p), f.jac(x, p)
end
function (f::ExplicitPolishJacobian{Inplace})(x, p)
    u = Vector{Float64}(undef, length(x))
    J = Matrix{Float64}(undef, length(x), length(x))
    f.pf(u, x, p)
    f.jac(J, x, p)
    return u, J
end
function (f::ExplicitPolishJacobian{Scalar})(x, p)
    u = Vector{Float64}(undef, 1)
    f.pf(u, x, p)
    return u, fill(f.jac(x[1], p), 1, 1)
end

function unpolynomialize_roots(f, roots, p, denominator_abstol, isscalar)
    validsols = isscalar ? Float64[] : Vector{Float64}[]
    for u in roots
        u_p = isscalar ? u[1] : u
        if any(<=(denominator_abstol) ∘ abs, f.denominator(u_p, p))
            continue
        end
        for sol in f.unpolynomialize(u_p, p)
            any(isnan, sol) && continue
            push!(validsols, sol)
        end
    end
    return validsols
end

function CommonSolve.solve(
        prob::NonlinearProblem, alg::TaylorHomotopyContinuationJL{true};
        abstol = 1.0e-10, maxiters = 200, denominator_abstol = 1.0e-7, kwargs...
    )
    u0 = state_values(prob)
    isscalar = u0 isa Number
    f, roots, _, orig_sol = taylor_homotopy_preprocessing(
        prob, alg; abstol, maxiters, kwargs...
    )

    validsols = unpolynomialize_roots(
        f, roots, parameter_values(prob),
        denominator_abstol, isscalar
    )

    if isempty(validsols)
        retcode = SciMLBase.ReturnCode.ConvergenceFailure
        resid = NonlinearSolveBase.Utils.evaluate_f(prob, u0)
        nlsol = SciMLBase.build_solution(prob, alg, u0, resid; retcode, original = orig_sol)
        return SciMLBase.EnsembleSolution([nlsol], 0.0, false, nothing)
    end

    retcode = SciMLBase.ReturnCode.Success
    nlsols = map(validsols) do u
        resid = NonlinearSolveBase.Utils.evaluate_f(prob, u)
        return SciMLBase.build_solution(prob, alg, u, resid; retcode, original = orig_sol)
    end
    return SciMLBase.EnsembleSolution(nlsols, 0.0, true, nothing)
end

function CommonSolve.solve(
        prob::NonlinearProblem, alg::TaylorHomotopyContinuationJL{false};
        abstol = 1.0e-10, maxiters = 200, denominator_abstol = 1.0e-7, kwargs...
    )
    u0 = state_values(prob)
    isscalar = u0 isa Number
    f, roots, expansion_point, orig_sol = taylor_homotopy_preprocessing(
        prob, alg; abstol, maxiters, kwargs...
    )

    validsols = unpolynomialize_roots(
        f, roots, parameter_values(prob),
        denominator_abstol, isscalar
    )

    if isempty(validsols)
        retcode = SciMLBase.ReturnCode.ConvergenceFailure
        resid = NonlinearSolveBase.Utils.evaluate_f(prob, u0)
        return SciMLBase.build_solution(prob, alg, u0, resid; retcode, original = orig_sol)
    end

    _, idx = findmin(validsols) do sol
        norm(sol .- u0)
    end
    u = validsols[idx]
    resid = NonlinearSolveBase.Utils.evaluate_f(prob, u)
    retcode = SciMLBase.ReturnCode.Success
    return SciMLBase.build_solution(prob, alg, u, resid; retcode, original = orig_sol)
end
