"""
    NewtonRaphson(; concrete_jac = nothing, linsolve = nothing, linesearch = nothing,
        precs = DEFAULT_PRECS, autodiff = nothing)

An implementation of PseudoTransient method that is used to solve steady state problems in
an accelerated manner. It uses an adaptive time-stepping to integrate an initial value of
nonlinear problem until sufficient accuracy in the desired steady-state is achieved to
switch over to Newton's method and gain a rapid convergence. This implementation
specifically uses "switched evolution relaxation" SER method.

This is all a fancy word soup for saying this is a Damped Newton Method with a scalar
damping value.

### Keyword Arguments

  - `autodiff`: determines the backend used for the Jacobian. Note that this argument is
    ignored if an analytical Jacobian is passed, as that will be used instead. Defaults to
    `nothing` which means that a default is selected according to the problem specification!
    Valid choices are types from ADTypes.jl.
  - `concrete_jac`: whether to build a concrete Jacobian. If a Krylov-subspace method is used,
    then the Jacobian will not be constructed and instead direct Jacobian-vector products
    `J*v` are computed using forward-mode automatic differentiation or finite differencing
    tricks (without ever constructing the Jacobian). However, if the Jacobian is still needed,
    for example for a preconditioner, `concrete_jac = true` can be passed in order to force
    the construction of the Jacobian.
  - `linsolve`: the [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) used for the
    linear solves within the Newton method. Defaults to `nothing`, which means it uses the
    LinearSolve.jl default algorithm choice. For more information on available algorithm
    choices, see the [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
  - `precs`: the choice of preconditioners for the linear solver. Defaults to using no
    preconditioners. For more information on specifying preconditioners for LinearSolve
    algorithms, consult the
    [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
  - `linesearch`: the line search algorithm to use. Defaults to [`NoLineSearch()`](@ref),
    which means that no line search is performed. Algorithms from `LineSearches.jl` must be
    wrapped in `LineSearchesJL` before being supplied.
  - `alpha_initial` : the initial pseudo time step. it defaults to 1e-3. If it is small,
    you are going to need more iterations to converge but it can be more stable.

### References

[1] Coffey, Todd S. and Kelley, C. T. and Keyes, David E. (2003), Pseudotransient
    Continuation and Differential-Algebraic Equations, SIAM Journal on Scientific Computing,
    25, 553-569. https://doi.org/10.1137/S106482750241044X
"""
function PseudoTransient(; concrete_jac = nothing, linsolve = nothing,
        linesearch::AbstractNonlinearSolveLineSearchAlgorithm = NoLineSearch(),
        precs = DEFAULT_PRECS, autodiff = nothing, alpha_initial = 1e-3)
    descent = DampedNewtonDescent(; linsolve, precs, initial_damping = alpha_initial,
        damping_fn = SwitchedEvolutionRelaxation())
    forward_ad = ifelse(autodiff isa ADTypes.AbstractForwardMode, autodiff, nothing)
    reverse_ad = ifelse(autodiff isa ADTypes.AbstractReverseMode, autodiff, nothing)

    return GeneralizedFirstOrderRootFindingAlgorithm{concrete_jac, :PseudoTransient}(linesearch,
        descent, autodiff, forward_ad, reverse_ad)
end

struct SwitchedEvolutionRelaxation <: AbstractDampingFunction end

@concrete mutable struct SwitchedEvolutionRelaxationCache <: AbstractDampingFunction
    res_norm
    α
    internalnorm
end

function SciMLBase.init(prob::AbstractNonlinearProblem, f::SwitchedEvolutionRelaxation,
        initial_damping, J, fu, u, args...; internalnorm::F = DEFAULT_NORM,
        kwargs...) where {F}
    T = promote_type(eltype(u), eltype(fu))
    return SwitchedEvolutionRelaxationCache(internalnorm(fu), T(initial_damping),
        internalnorm)
end

function SciMLBase.solve!(damping::SwitchedEvolutionRelaxationCache, J, fu, args...;
        kwargs...)
    res_norm = damping.internalnorm(fu)
    damping.α = cache.res_norm / res_norm
    cache.res_norm = res_norm
    return -damping.α * I
end
