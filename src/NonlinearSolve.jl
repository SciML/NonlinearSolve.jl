module NonlinearSolve

  include("utils.jl")
  include("bisection.jl")
  include("falsi.jl")

  export bisection, falsi
end # module
