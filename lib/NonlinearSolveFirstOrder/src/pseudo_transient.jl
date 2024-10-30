"""
    PseudoTransient(;
        concrete_jac = nothing, linesearch = missing, alpha_initial = 1e-3,
        linsolve = nothing, precs = nothing,
        autodiff = nothing, jvp_autodiff = nothing, vjp_autodiff = nothing
    )

An implementation of PseudoTransient Method [coffey2003pseudotransient](@cite) that is used
to solve steady state problems in an accelerated manner. It uses an adaptive time-stepping
to integrate an initial value of nonlinear problem until sufficient accuracy in the desired
steady-state is achieved to switch over to Newton's method and gain a rapid convergence.
This implementation specifically uses "switched evolution relaxation"
[kelley1998convergence](@cite) SER method.

### Keyword Arguments

  - `alpha_initial` : the initial pseudo time step. It defaults to `1e-3`. If it is small,
    you are going to need more iterations to converge but it can be more stable.
"""
function PseudoTransient(;
        concrete_jac = nothing, linesearch = missing, alpha_initial = 1e-3,
        linsolve = nothing, precs = nothing,
        autodiff = nothing, jvp_autodiff = nothing, vjp_autodiff = nothing
)
    return GeneralizedFirstOrderAlgorithm(;
        linesearch,
        descent = DampedNewtonDescent(;
            linsolve, precs, initial_damping = alpha_initial,
            damping_fn = SwitchedEvolutionRelaxation()
        ),
        autodiff,
        jvp_autodiff,
        vjp_autodiff,
        concrete_jac,
        name = :PseudoTransient
    )
end

"""
    SwitchedEvolutionRelaxation()

Method for updating the damping parameter in the [`PseudoTransient`](@ref) method based on
"switched evolution relaxation" [kelley1998convergence](@cite) SER method.
"""
struct SwitchedEvolutionRelaxation <: AbstractDampingFunction end

"""
    SwitchedEvolutionRelaxationCache <: AbstractDampingFunctionCache

Cache for the [`SwitchedEvolutionRelaxation`](@ref) method.
"""
@concrete mutable struct SwitchedEvolutionRelaxationCache <: AbstractDampingFunctionCache
    res_norm
    α⁻¹
    internalnorm
end

function NonlinearSolveBase.requires_normal_form_jacobian(::Union{
        SwitchedEvolutionRelaxation, SwitchedEvolutionRelaxationCache})
    return false
end

function NonlinearSolveBase.requires_normal_form_rhs(::Union{
        SwitchedEvolutionRelaxation, SwitchedEvolutionRelaxationCache})
    return false
end

function InternalAPI.init(
        prob::AbstractNonlinearProblem, f::SwitchedEvolutionRelaxation,
        initial_damping, J, fu, u, args...;
        internalnorm::F = L2_NORM, kwargs...
) where {F}
    T = promote_type(eltype(u), eltype(fu))
    return SwitchedEvolutionRelaxationCache(
        internalnorm(fu), T(inv(initial_damping)), internalnorm
    )
end

(damping::SwitchedEvolutionRelaxationCache)(::Nothing) = damping.α⁻¹

function InternalAPI.solve!(
        damping::SwitchedEvolutionRelaxationCache, J, fu, args...; kwargs...
)
    res_norm = damping.internalnorm(fu)
    damping.α⁻¹ *= res_norm / damping.res_norm
    damping.res_norm = res_norm
    return damping.α⁻¹
end
