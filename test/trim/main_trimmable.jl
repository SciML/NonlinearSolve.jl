using TrimTest

function (@main)(argv::Vector{String})::Cint
    λ = parse(Float64, argv[2])
    sol = TrimTest.TestModuleTrimmable.minimize(λ)
    println(Core.stdout, sum(sol.u))
    return 0
end
