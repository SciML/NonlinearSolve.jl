using NonlinearSolve

using SciMLBase

# Float32 must stay Float32 (no promotion through the augmented packing/predictor).
H(u, p, λ) = [(1 - λ) * (u[1] - p.c) + λ * (u[1]^2 - p.c)]
prob32 = HomotopyProblem(H, Float32[4.0], (c = 4.0f0,); λspan = (0.0f0, 1.0f0))
sol32 = solve(prob32, ArcLengthContinuation())
@test SciMLBase.successful_retcode(sol32)
@test eltype(sol32.u) == Float32
@test sol32.u[1] ≈ 2.0f0 atol = 1.0f-4

# in-place residual f(du, u, p, λ) is supported (the augmented corrector writes the n
# homotopy rows in place and appends the scalar arclength constraint itself).
function Hiip(du, u, p, λ)
    du[1] = (1 - λ) * (u[1] - 4.0) + λ * (u[1]^2 - 4.0)
    return nothing
end
probi = HomotopyProblem(NonlinearFunction{true}(Hiip), [4.0]; λspan = (0.0, 1.0))
soli = solve(probi, ArcLengthContinuation())
@test SciMLBase.successful_retcode(soli)
@test soli.u[1] ≈ 2.0 atol = 1.0e-6

# a multi-dimensional system tracks too (n = 2)
function H2(u, p, λ)
    return [
        (1 - λ) * (u[1] - 1.0) + λ * (u[1]^2 + u[2]^2 - 2.0),
        (1 - λ) * (u[2] - 1.0) + λ * (u[1] - u[2]),
    ]
end
prob2 = HomotopyProblem(H2, [1.0, 1.0]; λspan = (0.0, 1.0))
sol2 = solve(prob2, ArcLengthContinuation())
@test SciMLBase.successful_retcode(sol2)
@test sol2.u[1] ≈ 1.0 atol = 1.0e-6   # u1 = u2 and u1^2+u2^2 = 2 ⇒ u = (1, 1)
@test sol2.u[2] ≈ 1.0 atol = 1.0e-6
