module NonlinearSolve

  using Reexport
  using UnPack: @unpack
  using FiniteDiff, ForwardDiff
  using Setfield
  using StaticArrays
  using RecursiveArrayTools

  abstract type AbstractNonlinearProblem{uType,isinplace} end
  abstract type AbstractNonlinearSolveAlgorithm end
  abstract type AbstractBracketingAlgorithm <: AbstractNonlinearSolveAlgorithm end
  abstract type AbstractNewtonAlgorithm{CS,AD} <: AbstractNonlinearSolveAlgorithm end
  abstract type AbstractNonlinearSolver end
  abstract type AbstractImmutableNonlinearSolver <: AbstractNonlinearSolver end

  include("utils.jl")
  include("jacobian.jl")
  include("types.jl")
  include("solve.jl")
  include("bisection.jl")
  include("falsi.jl")
  include("raphson.jl")
  include("scalar.jl")

  # raw methods
  export bisection, falsi

  # DiffEq styled algorithms
  export Bisection, Falsi, NewtonRaphson

  export NonlinearProblem

  export solve, init, solve!

  export reinit!
end # module
