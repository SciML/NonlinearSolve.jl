using SciMLBase: NonlinearLeastSquaresProblem, NonlinearFunction

nonlinear_functions = (
    (NonlinearFunction{false}((u, p) -> (u .^ 2 .- p)[1:1]), [0.1, 0.0]),
    (NonlinearFunction{false}((u, p) -> vcat(u .* u .- p, u .* u .- p)), [0.1, 0.1]),
    (
        NonlinearFunction{true}(
            (du, u, p) -> du[1] = u[1] * u[1] - p, resid_prototype = zeros(1)
        ),
        [0.1, 0.0]
    ),
    (
        NonlinearFunction{true}(
            (du, u, p) -> du .= vcat(u .* u .- p, u .* u .- p), resid_prototype = zeros(4)
        ),
        [0.1, 0.1]
    )
)

nlls_problems = NonlinearLeastSquaresProblem[]
for (fn, u0) in nonlinear_functions
    push!(nlls_problems, NonlinearLeastSquaresProblem(fn, u0, 2.0))
end
