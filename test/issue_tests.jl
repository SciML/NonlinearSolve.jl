@testitem "Correct Best Solution: #565" tags = [:core] begin
    using NonlinearSolve, StableRNGs

    x = collect(0:0.1:10)

    line_fct(x, p) = p[1] .+ p[2] .* x

    y_line = line_fct(x, [1, 3])
    y_line_n = line_fct(x, [1, 3]) + randn(StableRNG(0), length(x))

    res(β, (x, y)) = line_fct(x, β) .- y

    prob = NonlinearLeastSquaresProblem(res, [1, 3], p = (x, y_line_n))
    sol1 = solve(prob; maxiters = 1000)

    prob = NonlinearLeastSquaresProblem(res, [1, 5], p = (x, y_line_n))
    sol2 = solve(prob; maxiters = 1000)

    @test sol1.u ≈ sol2.u
end
