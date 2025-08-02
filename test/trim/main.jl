using TrimTest

function (@main)(argv::Vector{String})::Cint
    λ = try
        parse(Float64, argv[2])
    catch
        parse(Float64, argv[1])
    end
    sol = TrimTest.minimize(λ)
    println(Core.stdout, sum(sol.u))
    return 0
end
