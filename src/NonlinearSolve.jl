module NonlinearSolve

  using Reexport
  @reexport using DiffEqBase
  using UnPack: @unpack
  using FiniteDiff, ForwardDiff
  using Setfield
  using StaticArrays

  abstract type AbstractNonlinearSolveAlgorithm end
  abstract type AbstractBracketingAlgorithm <: AbstractNonlinearSolveAlgorithm end
  abstract type AbstractNewtonAlgorithm{CS,AD} <: AbstractNonlinearSolveAlgorithm end
  abstract type AbstractNonlinearSolver end
  abstract type AbstractImmutableNonlinearSolver <: AbstractNonlinearSolver end

  include("jacobian.jl")
  include("types.jl")
  include("utils.jl")
  include("solve.jl")
  include("bisection.jl")
  include("falsi.jl")
  include("raphson.jl")
  include("scalar.jl")

  # raw methods
  export bisection, falsi

  # DiffEq styled algorithms
  export Bisection, Falsi, NewtonRaphson

  export reinit!
end # module
