module NonlinearSolve

  using Reexport
  @reexport using DiffEqBase
  using UnPack: @unpack

  abstract type AbstractNonlinearSolveAlgorithm end
  abstract type AbstractBracketingAlgorithm <: AbstractNonlinearSolveAlgorithm end

  include("types.jl")
  include("solve.jl")
  include("utils.jl")
  include("bisection.jl")
  include("falsi.jl")

  # raw methods
  export bisection, falsi

  # DiffEq styled algorithms
  export Bisection, Falsi
end # module
