using TrimTest

function (@main)(argv::Vector{String})::Cint
    λ = parse(Float64, argv[1])
    sol = TrimTest.TestModuleClean.minimize(λ)
    println(Core.stdout, sum(sol.u))
    return 0
end
