using Pkg
using SafeTestsets
const LONGER_TESTS = false

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV,"APPVEYOR")

@time begin

if GROUP == "All" || GROUP == "Interface"
  #@time @safetestset "Linear Solver Tests" begin include("interface/linear_solver_test.jl") end
  @time @safetestset "Basic Tests + Some AD" begin include("basictests.jl") end
end

end
