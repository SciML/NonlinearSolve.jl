"""
    HomotopySweep(; inner = nothing, nsteps = 10, adaptive = true, min_dλ = 1e-3)

Natural-parameter continuation solver for a [`SciMLBase.HomotopyProblem`](@ref). The
continuation parameter ``λ`` is swept across the problem's `λspan` in `nsteps` steps; each
step solves the inner nonlinear system with `inner`, warm-started from the previous step's
solution. When `adaptive` is `true`, a step whose inner solve fails to converge halves the
λ increment and retries, down to a floor of `min_dλ`.

`inner` is the inner nonlinear algorithm; `nothing` selects NonlinearSolve's default
polyalgorithm (NOT a hardcoded Newton). This is the embedding-homotopy / continuation
analogue used to robustly initialize systems whose target form is hard to solve cold; it is
unrelated to the polynomial `HomotopyContinuationJL`.
"""
@concrete struct HomotopySweep <: AbstractNonlinearSolveAlgorithm
    inner
    nsteps::Int
    adaptive::Bool
    min_dλ
end

function HomotopySweep(; inner = nothing, nsteps = 10, adaptive = true, min_dλ = 1.0e-3)
    nsteps >= 1 || throw(ArgumentError("HomotopySweep `nsteps` must be ≥ 1, got $nsteps"))
    return HomotopySweep(inner, nsteps, adaptive, min_dλ)
end

# Build a pure setter `(p, λ) -> p′` that writes λ at the problem's homotopy_parameter.
# Integer locator → positional index into a copy of `p`.
# Symbolic locator → setp_oop, which yields a fresh parameter container with λ set.
function _lambda_setter(prob::HomotopyProblem, loc::Integer)
    return (p, λ) -> begin
        T = promote_type(eltype(p), typeof(λ))
        p2 = similar(p, T)
        copyto!(p2, p)
        p2[loc] = λ
        return p2
    end
end

function _lambda_setter(prob::HomotopyProblem, loc)
    # setp_oop returns a setter `(p, λ) -> new_p` that yields a fresh parameter container
    # with λ set, without mutating `p` — the correct out-of-place SII contract.
    setter = SymbolicIndexingInterface.setp_oop(prob, loc)
    return (p, λ) -> setter(p, λ)
end

function CommonSolve.solve(
        prob::HomotopyProblem, alg::HomotopySweep, args...; kwargs...)
    loc = prob.homotopy_parameter
    loc === nothing && throw(ArgumentError(
        "HomotopyProblem.homotopy_parameter is `nothing`; HomotopySweep needs to know " *
        "which parameter is λ. Construct the problem with `homotopy_parameter = <index or symbol>`."))
    set_λ = _lambda_setter(prob, loc)

    λ0, λ1 = prob.λspan
    u = copy(prob.u0)
    dλ = (λ1 - λ0) / alg.nsteps
    λ = float(λ0)
    local last_sol

    while true
        next_λ = abs(λ1 - λ) <= abs(dλ) ? λ1 : λ + dλ
        inner_prob = NonlinearProblem(prob.f, u, set_λ(prob.p, next_λ))
        last_sol = solve(inner_prob, alg.inner, args...; kwargs...)

        if SciMLBase.successful_retcode(last_sol)
            u = last_sol.u
            λ = next_λ
            λ == λ1 && break
        elseif alg.adaptive && abs(dλ) / 2 >= alg.min_dλ
            # conservative: step size is not restored after a bisection
            dλ = dλ / 2          # bisect; retry from the same λ (do not advance)
        else
            # on failure: u is the last CONVERGED iterate (λ<λ1); resid is from the failed step (advisory)
            return SciMLBase.build_solution(
                prob, alg, u, last_sol.resid; retcode = last_sol.retcode)
        end
    end

    return SciMLBase.build_solution(
        prob, alg, u, last_sol.resid; retcode = ReturnCode.Success)
end
