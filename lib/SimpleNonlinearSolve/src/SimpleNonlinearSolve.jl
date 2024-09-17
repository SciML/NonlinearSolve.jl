module SimpleNonlinearSolve

using ADTypes: ADTypes, AbstractADType, AutoFiniteDiff, AutoForwardDiff,
               AutoPolyesterForwardDiff
using PrecompileTools: @compile_workload, @setup_workload
using Reexport: @reexport
@reexport using SciMLBase  # I don't like this but needed to avoid a breaking change

using BracketingNonlinearSolve: Alefeld, Bisection, Brent, Falsi, ITP, Ridder

@setup_workload begin
    @compile_workload begin end
end

export AutoFiniteDiff, AutoForwardDiff, AutoPolyesterForwardDiff

export Alefeld, Bisection, Brent, Falsi, ITP, Ridder

end
