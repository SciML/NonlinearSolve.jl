"""
    NewtonRaphson(; concrete_jac = nothing, linsolve = nothing,
        linesearch::AbstractNonlinearSolveLineSearchAlgorithm = NoLineSearch(),
        precs = DEFAULT_PRECS, autodiff = nothing)

An implementation of PseudoTransient method that is used to solve steady state problems in
an accelerated manner. It uses an adaptive time-stepping to integrate an initial value of
nonlinear problem until sufficient accuracy in the desired steady-state is achieved to
switch over to Newton's method and gain a rapid convergence. This implementation
specifically uses "switched evolution relaxation" SER method.

### Keyword Arguments

  - `alpha_initial` : the initial pseudo time step. it defaults to 1e-3. If it is small,
    you are going to need more iterations to converge but it can be more stable.

### References

[1] Kelley, Carl Timothy, and David E. Keyes. "Convergence analysis of pseudo-transient
continuation." SIAM Journal on Numerical Analysis 35.2 (1998): 508-523.
[2] Coffey, Todd S. and Kelley, C. T. and Keyes, David E. (2003), Pseudotransient
Continuation and Differential-Algebraic Equations, SIAM Journal on Scientific Computing,
25, 553-569. https://doi.org/10.1137/S106482750241044X
"""
function PseudoTransient(; concrete_jac = nothing, linsolve = nothing,
        linesearch::AbstractNonlinearSolveLineSearchAlgorithm = NoLineSearch(),
        precs = DEFAULT_PRECS, autodiff = nothing, alpha_initial = 1e-3)
    descent = DampedNewtonDescent(; linsolve, precs, initial_damping = alpha_initial,
        damping_fn = SwitchedEvolutionRelaxation())
    return GeneralizedFirstOrderAlgorithm(; concrete_jac,
        name = :PseudoTransient, linesearch, descent, jacobian_ad = autodiff)
end

struct SwitchedEvolutionRelaxation <: AbstractDampingFunction end

@concrete mutable struct SwitchedEvolutionRelaxationCache <: AbstractDampingFunctionCache
    res_norm
    α⁻¹
    internalnorm
end

function requires_normal_form_jacobian(cache::Union{SwitchedEvolutionRelaxation,
        SwitchedEvolutionRelaxationCache})
    return false
end
function requires_normal_form_rhs(cache::Union{SwitchedEvolutionRelaxation,
        SwitchedEvolutionRelaxationCache})
    return false
end

function SciMLBase.init(prob::AbstractNonlinearProblem, f::SwitchedEvolutionRelaxation,
        initial_damping, J, fu, u, args...; internalnorm::F = DEFAULT_NORM,
        kwargs...) where {F}
    T = promote_type(eltype(u), eltype(fu))
    return SwitchedEvolutionRelaxationCache(internalnorm(fu), T(1 / initial_damping),
        internalnorm)
end

(damping::SwitchedEvolutionRelaxationCache)(::Nothing) = damping.α⁻¹

function SciMLBase.solve!(damping::SwitchedEvolutionRelaxationCache, J, fu, args...;
        kwargs...)
    res_norm = damping.internalnorm(fu)
    damping.α⁻¹ *= res_norm / damping.res_norm
    damping.res_norm = res_norm
    return damping.α⁻¹
end
