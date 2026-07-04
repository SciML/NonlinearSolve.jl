"""
    PseudoTransient(;
        concrete_jac = nothing, linesearch = missing, alpha_initial = 1e-3,
        linsolve = nothing, mass_matrix = nothing,
        autodiff = nothing, jvp_autodiff = nothing, vjp_autodiff = nothing
    )

An implementation of PseudoTransient Method [coffey2003pseudotransient](@cite) that is used
to solve steady state problems in an accelerated manner. It uses an adaptive time-stepping
to integrate an initial value of nonlinear problem until sufficient accuracy in the desired
steady-state is achieved to switch over to Newton's method and gain a rapid convergence.
This implementation specifically uses "switched evolution relaxation"
[kelley1998convergence](@cite) SER method.

The damped Newton step solves ``(J(u) + (1/α) M) δu = -F(u)``, i.e. one implicit-Euler step
of the fictitious dynamics ``M u' = -F(u)`` with pseudo-timestep `α`. By default `M = I`,
which recovers the classical pseudo-transient continuation. Supplying a mass matrix `M`
generalizes this to differential-algebraic (DAE) steady-state problems
[coffey2003pseudotransient](@cite), where `M` is the (possibly singular, structured) mass
matrix of the underlying `M u' = -F(u)` system. This makes the continuation topology-aware:
components with large `M` entries are damped less, algebraic components (zero rows of `M`)
are treated consistently with the DAE structure.

### Keyword Arguments

  - `alpha_initial` : the initial pseudo time step. It defaults to `1e-3`. If it is small,
    you are going to need more iterations to converge but it can be more stable.
  - `mass_matrix` : the mass matrix `M` used for damping, i.e. the descent solves
    ``(J + (1/α) M) δu = -F``. Defaults to `nothing`, which uses `M = I` (identity damping,
    bit-for-bit identical to the classical method). If `nothing` and the problem's
    `NonlinearFunction` carries a non-identity `mass_matrix`, that mass matrix is used
    automatically. A diagonal `M` (e.g. `Diagonal(...)`) uses an efficient diagonal update;
    a general sparse/dense `M` is supported as well. Intended for square DAE-derived systems.
"""
function PseudoTransient(;
        concrete_jac = nothing, linesearch = missing, alpha_initial = 1.0e-3,
        linsolve = nothing, mass_matrix = nothing,
        autodiff = nothing, jvp_autodiff = nothing, vjp_autodiff = nothing
    )
    return GeneralizedFirstOrderAlgorithm(;
        linesearch,
        descent = DampedNewtonDescent(;
            linsolve, initial_damping = alpha_initial,
            damping_fn = SwitchedEvolutionRelaxation(mass_matrix)
        ),
        autodiff,
        jvp_autodiff,
        vjp_autodiff,
        concrete_jac,
        name = :PseudoTransient
    )
end

"""
    SwitchedEvolutionRelaxation(mass_matrix = nothing)

Method for updating the damping parameter in the [`PseudoTransient`](@ref) method based on
"switched evolution relaxation" [kelley1998convergence](@cite) SER method.

The optional `mass_matrix` generalizes the damping term from the identity `I` to a
user-supplied (constant) matrix `M`, so that the damped Newton step becomes
``(J + (1/α) M) δu = -F``. When `mass_matrix === nothing`, the damping is the scalar
`1/α` applied to the diagonal, recovering the classical identity-damped method exactly.
"""
@concrete struct SwitchedEvolutionRelaxation <: AbstractDampingFunction
    mass_matrix
end

SwitchedEvolutionRelaxation() = SwitchedEvolutionRelaxation(nothing)

"""
    SwitchedEvolutionRelaxationCache <: AbstractDampingFunctionCache

Cache for the [`SwitchedEvolutionRelaxation`](@ref) method.
"""
@concrete mutable struct SwitchedEvolutionRelaxationCache <: AbstractDampingFunctionCache
    res_norm
    α⁻¹
    internalnorm
    mass_matrix
    D
end

function NonlinearSolveBase.requires_normal_form_jacobian(
        ::Union{
            SwitchedEvolutionRelaxation, SwitchedEvolutionRelaxationCache,
        }
    )
    return false
end

function NonlinearSolveBase.requires_normal_form_rhs(
        ::Union{
            SwitchedEvolutionRelaxation, SwitchedEvolutionRelaxationCache,
        }
    )
    return false
end

function InternalAPI.init(
        prob::AbstractNonlinearProblem, f::SwitchedEvolutionRelaxation,
        initial_damping, J, fu, u, args...;
        internalnorm::F = L2_NORM, kwargs...
    ) where {F}
    T = promote_type(eltype(u), eltype(fu))
    α⁻¹ = T(inv(initial_damping))
    M = resolve_ser_mass_matrix(f.mass_matrix, prob)
    D = M === nothing ? nothing : α⁻¹ .* M
    return SwitchedEvolutionRelaxationCache(internalnorm(fu), α⁻¹, internalnorm, M, D)
end

# Resolve the mass matrix used for damping: an explicit one always wins, otherwise fall
# back to the problem's `NonlinearFunction` mass matrix if it is a genuine (non-identity)
# matrix. `I` / `UniformScaling` / `nothing` all map back to scalar identity damping so the
# classical behavior is preserved bit-for-bit.
resolve_ser_mass_matrix(mass_matrix, ::AbstractNonlinearProblem) = mass_matrix
function resolve_ser_mass_matrix(::Nothing, prob::AbstractNonlinearProblem)
    hasproperty(prob.f, :mass_matrix) || return nothing
    M = prob.f.mass_matrix
    (M === nothing || M isa LinearAlgebra.UniformScaling) && return nothing
    return M
end

(damping::SwitchedEvolutionRelaxationCache)(::Nothing) =
    damping.mass_matrix === nothing ? damping.α⁻¹ : damping.D

function InternalAPI.solve!(
        damping::SwitchedEvolutionRelaxationCache, J, fu, args...; kwargs...
    )
    res_norm = damping.internalnorm(fu)
    damping.α⁻¹ *= res_norm / damping.res_norm
    damping.res_norm = res_norm
    damping.mass_matrix === nothing && return damping.α⁻¹
    damping.D = scale_mass_matrix!!(damping.D, damping.α⁻¹, damping.mass_matrix)
    return damping.D
end

scale_mass_matrix!!(::Nothing, α⁻¹, M) = α⁻¹ .* M
function scale_mass_matrix!!(D, α⁻¹, M)
    ArrayInterface.can_setindex(D) || return α⁻¹ .* M
    @. D = α⁻¹ * M
    return D
end
