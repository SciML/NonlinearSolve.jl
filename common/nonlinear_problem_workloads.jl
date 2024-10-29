using SciMLBase: NonlinearProblem, NonlinearFunction, NoSpecialize

nonlinear_functions = (
    (NonlinearFunction{false, NoSpecialize}((u, p) -> u .* u .- p), 0.1),
    (NonlinearFunction{false, NoSpecialize}((u, p) -> u .* u .- p), [0.1]),
    (NonlinearFunction{true, NoSpecialize}((du, u, p) -> du .= u .* u .- p), [0.1])
)

nonlinear_problems = NonlinearProblem[]
for (fn, u0) in nonlinear_functions
    push!(nonlinear_problems, NonlinearProblem(fn, u0, 2.0))
end
