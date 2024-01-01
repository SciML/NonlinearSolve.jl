
# struct SwitchedEvolutionRelaxation <: AbstractDampingFunction end

# @concrete mutable struct SwitchedEvolutionRelaxationCache <: AbstractDampingFunctionCache
#     res_norm
#     α
#     internalnorm
# end

# function SciMLBase.init(prob::AbstractNonlinearProblem, f::SwitchedEvolutionRelaxation,
#         initial_damping, J, fu, u, args...; internalnorm::F = DEFAULT_NORM,
#         kwargs...) where {F}
#     T = promote_type(eltype(u), eltype(fu))
#     return SwitchedEvolutionRelaxationCache(internalnorm(fu), T(initial_damping),
#         internalnorm)
# end

# function SciMLBase.solve!(damping::SwitchedEvolutionRelaxationCache, J, fu, args...;
#         kwargs...)
#     res_norm = damping.internalnorm(fu)
#     damping.α = damping.res_norm / res_norm
#     damping.res_norm = res_norm
#     return damping.α
# end
