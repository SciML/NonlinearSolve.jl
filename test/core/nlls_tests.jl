
# TODO: Test polyalg here

@testitem "NLLS Analytic Jacobian" tags=[:core] begin
    dataIn = 1:10
    f(x, p) = x[1] * dataIn .^ 2 .+ x[2] * dataIn .+ x[3]
    dataOut = f([1, 2, 3], nothing) + 0.1 * randn(10, 1)

    resid(x, p) = f(x, p) - dataOut
    jac(x, p) = [dataIn .^ 2 dataIn ones(10, 1)]
    x0 = [1, 1, 1]

    prob = NonlinearLeastSquaresProblem(resid, x0)
    sol1 = solve(prob)

    nlfunc = NonlinearFunction(resid; jac)
    prob = NonlinearLeastSquaresProblem(nlfunc, x0)
    sol2 = solve(prob)

    @test sol1.u ≈ sol2.u
end
