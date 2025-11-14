module MyModule
include("./optimization_once_per_process.jl")
end

function (@main)(argv::Vector{String})::Cint
    λ = parse(Float64, argv[1])
    sol = MyModule.TestModuleTrimmable.minimize(λ)
    println(Core.stdout, sum(sol.u))
    return 0
end
