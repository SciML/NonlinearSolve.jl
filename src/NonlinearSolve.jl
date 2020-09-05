module NonlinearSolve

  using Reexport
  @reexport using DiffEqBase
  using UnPack: @unpack
  using FiniteDiff, ForwardDiff

  abstract type AbstractNonlinearSolveAlgorithm end
  abstract type AbstractBracketingAlgorithm <: AbstractNonlinearSolveAlgorithm end
  abstract type AbstractNewtonAlgorithm{CS,AD} <: AbstractNonlinearSolveAlgorithm end
  abstract type AbstractNonlinearSolver end

  include("jacobian.jl")
  include("types.jl")
  include("solve.jl")
  include("utils.jl")
  include("bisection.jl")
  include("falsi.jl")
  include("raphson.jl")

  # raw methods
  export bisection, falsi

  # DiffEq styled algorithms
  export Bisection, Falsi, NewtonRaphson
end # module
