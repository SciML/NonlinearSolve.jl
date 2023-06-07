using NonlinearSolve, NLsolve, LinearAlgebra

# Implementation of the 23 test problems in
# [test_nonlin](https://people.sc.fsu.edu/~jburkardt/m_src/test_nonlin/test_nonlin.html)

# ------------------------------------- Problem 1 ------------------------------------------
function p1_f!(out, x, p = nothing)
    n = length(x)
    out[1] = 1.0 - x[1]
    out[2:n] .= 10.0 .* (x[2:n] .- x[1:(n - 1)] .* x[1:(n - 1)])
    nothing
end

n = 10
x_sol = ones(n)
x_start = ones(n)
x_start[1] = -1.2
p1_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Generalized Rosenbrock function")

# ------------------------------------- Problem 2 ------------------------------------------
function p2_f!(out, x, p = nothing)
    out[1] = x[1] + 10.0 * x[2]
    out[2] = sqrt(5.0) * (x[3] - x[4])
    out[3] = (x[2] - 2.0 * x[3])^2
    out[4] = sqrt(10.0) * (x[1] - x[4]) * (x[1] - x[4])
    nothing
end

n = 4
x_sol = zeros(n)
x_start = [3.0, -1.0, 0.0, 1.0]
p2_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Powell singular function")

# ------------------------------------- Problem 3 ------------------------------------------
function p3_f!(out, x, p = nothing)
    out[1] = 10000.0 * x[1] * x[2] - 1.0
    out[2] = exp(-x[1]) + exp(-x[2]) - 1.0001
    nothing
end

n = 2
x_sol = [1.098159e-05, 9.106146]
x_start = [0.0, 1.0]
p3_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Powell badly scaled function")

# ------------------------------------- Problem 4 ------------------------------------------
function p4_f!(out, x, p = nothing)
    temp1 = x[2] - x[1] * x[1]
    temp2 = x[4] - x[3] * x[3]

    out[1] = -200.0 * x[1] * temp1 - (1.0 - x[1])
    out[2] = 200.0 * temp1 + 20.2 * (x[2] - 1.0) + 19.8 * (x[4] - 1.0)
    out[3] = -180.0 * x[3] * temp2 - (1.0 - x[3])
    out[4] = 180.0 * temp2 + 20.2 * (x[4] - 1.0) + 19.8 * (x[2] - 1.0)
    nothing
end

n = 4
x_sol = ones(n)
x_start = [-3.0, -1.0, -3.0, -1.0]
p4_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Wood function")

# ------------------------------------- Problem 5 ------------------------------------------
function p5_f!(out, x, p = nothing)
    if 0.0 < x[1]
        temp = atan(x[2] / x[1]) / (2.0 * pi)
    elseif x[1] < 0.0
        temp = atan(x[2] / x[1]) / (2.0 * pi) + 0.5
    else
        temp = 0.25 * sign(x[2])
    end

    out[1] = 10.0 * (x[3] - 10.0 * temp)
    out[2] = 10.0 * (sqrt(x[1] * x[1] + x[2] * x[2]) - 1.0)
    out[3] = x[3]
    nothing
end

n = 3
x_sol = [1.0, 0.0, 0.0]
x_start = [-1.0, 0.0, 0.0]
p5_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Helical valley function")

# ------------------------------------- Problem 6 ------------------------------------------
function p6_f!(out, x, p = nothing)
    n = length(x)
    for i in 1:29
        ti = i / 29.0
        sum1 = 0.0
        temp = 1.0
        for j in 2:n
            sum1 = sum1 + j * temp * x[j]
            temp = ti * temp
        end

        sum2 = 0.0
        temp = 1.0
        for j in 1:n
            sum2 = sum2 + temp * x[j]
            temp = ti * temp
        end
        temp = 1.0 / ti

        for k in 1:n
            out[k] = out[k] + temp * (sum1 - sum2 * sum2 - 1.0) * (k - 2.0 * ti * sum2)
            temp = ti * temp
        end
    end

    out[1] = out[1] + 3.0 * x[1] - 2.0 * x[1] * x[1] + 2.0 * x[1]^3
    out[2] = out[2] + x[2] - x[2]^2 - 1.0
    nothing
end

n = 2
x_sol = []
x_start = zeros(n)
p6_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Watson function")

# ------------------------------------- Problem 7 ------------------------------------------
function p7_f!(out, x, p = nothing)
    n = length(x)
    out .= 0.0
    for j in 1:n
        t1 = 1.0
        t2 = x[j]
        for i in 1:n
            out[i] += t2
            t3 = 2.0 * x[j] * t2 - t1
            t1 = t2
            t2 = t3
        end
    end
    out ./= n

    for i in 1:n
        ip1 = i
        if ip1 % 2 == 0
            out[i] = out[i] + 1.0 / (ip1 * ip1 - 1)
        end
    end
    nothing
end

n = 2
x_sol = [0.2113248654051871, 0.7886751345948129]
x_sol .= 2.0 .* x_sol .- 1.0
x_start = zeros(n)
for i in 1:n
    x_start[i] = (2 * i - n) / (n + 1)
end
p7_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Chebyquad function")

# ------------------------------------- Problem 8 ------------------------------------------
function p8_f!(out, x, p = nothing)
    n = length(x)
    out[1:(n - 1)] .= x[1:(n - 1)] .+ sum(x) .- (n + 1)
    out[n] = prod(x) - 1.0
    nothing
end

n = 10
x_sol = ones(n)
x_start = ones(n) ./ 2
p8_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Brown almost linear function")

# ------------------------------------- Problem 9 ------------------------------------------
function p9_f!(out, x, p = nothing)
    n = length(x)
    h = 1.0 / (n + 1)
    for k in 1:n
        out[k] = 2.0 * x[k] + 0.5 * h^2 * (x[k] + k * h + 1.0)^3
        if 1 < k
            out[k] = out[k] - x[k - 1]
        end
        if k < n
            out[k] = out[k] - x[k + 1]
        end
    end
    nothing
end

n = 10
x_sol = []
x_start = ones(n)
for i in 1:n
    x_start[i] = (i * (i - n - 1)) / (n + 1)^2
end
p9_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Discrete boundary value function")

# ------------------------------------- Problem 10 -----------------------------------------
function p10_f!(out, x, p = nothing)
    n = length(x)
    h = 1.0 / (n + 1)
    for k in 1:n
        tk = k / (n + 1)
        sum1 = 0.0
        for j in 1:k
            tj = j * h
            sum1 = sum1 + tj * (x[j] + tj + 1.0)^3
        end
        sum2 = 0.0
        for j in k:n
            tj = j * h
            sum2 = sum2 + (1.0 - tj) * (x[j] + tj + 1.0)^3
        end

        out[k] = x[k] + h * ((1.0 - tk) * sum1 + tk * sum2) / 2.0
    end
    nothing
end

n = 10
x_sol = []
x_start = zeros(n)
for i in 1:n
    x_start[i] = (i * (i - n - 1)) / (n + 1)^2
end
p10_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Discrete integral equation function")

# ------------------------------------- Problem 11 -----------------------------------------
function p11_f!(out, x, p = nothing)
    n = length(x)
    c_sum = sum(cos.(x))
    for k in 1:n
        out[k] = n - c_sum + k * (1.0 - cos(x[k])) - sin(x[k])
    end
    nothing
end

n = 10
x_sol = []
x_start = ones(n) / n
p11_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Trigonometric function")

# ------------------------------------- Problem 12 -----------------------------------------
function p12_f!(out, x, p = nothing)
    n = length(x)
    sum1 = 0.0
    for j in 1:n
        sum1 += j * (x[j] - 1.0)
    end
    for k in 1:n
        out[k] = x[k] - 1.0 + k * sum1 * (1.0 + 2.0 * sum1 * sum1)
    end
    nothing
end

n = 10
x_sol = ones(n)
x_start = zeros(n)
for i in 1:n
    x_start[i] = 1.0 - i / n
end
p12_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Variably dimensioned function")

# ------------------------------------- Problem 13 -----------------------------------------
function p13_f!(out, x, p = nothing)
    n = length(x)
    for k in 1:n
        out[k] = (3.0 - 2.0 * x[k]) * x[k] + 1.0
        if 1 < k
            out[k] -= x[k - 1]
        end
        if k < n
            out[k] -= 2.0 * x[k + 1]
        end
    end
    nothing
end

n = 10
x_sol = []
x_start = ones(n) .* (-1.0)
p13_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Broyden tridiagonal function")

# ------------------------------------- Problem 14 -----------------------------------------
function p14_f!(out, x, p = nothing)
    n = length(x)
    ml = 5
    mu = 1
    for k in 1:n
        k1 = max(1, k - ml)
        k2 = min(n, k + mu)

        temp = 0.0
        for j in k1:k2
            if j != k
                temp += x[j] * (1.0 + x[j])
            end
        end
        out[k] = x[k] * (2.0 + 5.0 * x[k] * x[k]) + 1.0 - temp
    end
    nothing
end

n = 10
x_sol = []
x_start = ones(n) .* (-1.0)
p14_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Broyden banded function")

# ------------------------------------- Problem 15 -----------------------------------------
function p15_f!(out, x, p = nothing)
    out[1] = (x[1] * x[1] + x[2] * x[3]) - 0.0001
    out[2] = (x[1] * x[2] + x[2] * x[4]) - 1.0
    out[3] = (x[3] * x[1] + x[4] * x[3]) - 0.0
    out[4] = (x[3] * x[2] + x[4] * x[4]) - 0.0001
    nothing
end

n = 4
x_sol = [0.01, 50.0, 0.0, 0.01]
x_start = [1.0, 0.0, 0.0, 1.0]
p15_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Hammarling 2 by 2 matrix square root problem")

# ------------------------------------- Problem 16 -----------------------------------------
function p16_f!(out, x, p = nothing)
    out[1] = (x[1] * x[1] + x[2] * x[4] + x[3] * x[7]) - 0.0001
    out[2] = (x[1] * x[2] + x[2] * x[5] + x[3] * x[8]) - 1.0
    out[3] = x[1] * x[3] + x[2] * x[6] + x[3] * x[9]
    out[4] = x[4] * x[1] + x[5] * x[4] + x[6] * x[7]
    out[5] = (x[4] * x[2] + x[5] * x[5] + x[6] * x[8]) - 0.0001
    out[6] = x[4] * x[3] + x[5] * x[6] + x[6] * x[9]
    out[7] = x[7] * x[1] + x[8] * x[4] + x[9] * x[7]
    out[8] = x[7] * x[2] + x[8] * x[5] + x[9] * x[8]
    out[9] = (x[7] * x[3] + x[8] * x[6] + x[9] * x[9]) - 0.0001
    nothing
end

n = 9
x_sol = [0.01, 50.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
x_start = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
p16_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Hammarling 3 by 3 matrix square root problem")

# ------------------------------------- Problem 17 -----------------------------------------
function p17_f!(out, x, p = nothing)
    out[1] = x[1] + x[2] - 3.0
    out[2] = x[1]^2 + x[2]^2 - 9.0
    nothing
end

n = 2
x_sol = [0.0, 3.0]
x_start = [1.0, 5.0]
p17_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Dennis and Schnabel 2 by 2 example")

# ------------------------------------- Problem 18 -----------------------------------------
function p18_f!(out, x, p = nothing)
    if x[1] != 0.0
        out[1] = x[2]^2 * (1.0 - exp(-x[1] * x[1])) / x[1]
    else
        out[1] = 0.0
    end
    if x[2] != 0.0
        out[2] = x[1] * (1.0 - exp(-x[2] * x[2])) / x[2]
    else
        out[2] = 0.0
    end
    nothing
end

n = 2
x_sol = zeros(n)
x_start = [2.0, 2.0]
p18_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Sample problem 18")

# ------------------------------------- Problem 19 -----------------------------------------
function p19_f!(out, x, p = nothing)
    out[1] = x[1] * (x[1]^2 + x[2]^2)
    out[2] = x[2] * (x[1]^2 + x[2]^2)
    nothing
end

n = 2
x_sol = zeros(n)
x_start = [3.0, 3.0]
p19_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Sample problem 19")

# ------------------------------------- Problem 20 -----------------------------------------
function p20_f!(out, x, p = nothing)
    out[1] = x[1] * (x[1] - 5.0)^2
    nothing
end

n = 1
x_sol = [5.0] # OR [0.0]...
x_start = [1.0]
p20_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Scalar problem f(x) = x(x - 5)^2")

# ------------------------------------- Problem 21 -----------------------------------------
function p21_f!(out, x, p = nothing)
    out[1] = x[1] - x[2]^3 + 5.0 * x[2]^2 - 2.0 * x[2] - 13.0
    out[2] = x[1] + x[2]^3 + x[2]^2 - 14.0 * x[2] - 29.0
    nothing
end

n = 2
x_sol = [5.0, 4.0]
x_start = [0.5, -2.0]
p21_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Freudenstein-Roth function")

# ------------------------------------- Problem 22 -----------------------------------------
function p22_f!(out, x, p = nothing)
    out[1] = x[1] * x[1] - x[2] + 1.0
    out[2] = x[1] - cos(0.5 * pi * x[2])
    nothing
end

n = 2
x_sol = [0.0, 1.0]
x_start = [1.0, 0.0]
p22_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Boggs function")

# ------------------------------------- Problem 23 -----------------------------------------
function p23_f!(out, x, p = nothing)
    c = 0.9
    out[1:n] = x[1:n]
    μ = zeros(n)
    for i in 1:n
        μ[i] = (2 * i) / (2 * n)
    end
    for i in 1:n
        s = 0.0
        for j in 1:n
            s = s + (μ[i] * x[j]) / (μ[i] + μ[j])
        end
        term = 1.0 - c * s / (2 * n)
        out[i] -= 1.0 / term
    end
    nothing
end

n = 10
x_sol = []
x_start = ones(n)
p23_dict = Dict("n" => n, "start" => x_start, "sol" => x_sol,
    "title" => "Chandrasekhar function")

# ----------------------------------- Solve problems ---------------------------------------
problems = (p1_f!, p2_f!, p3_f!, p4_f!, p5_f!, p6_f!, p7_f!, p8_f!, p9_f!, p10_f!, p11_f!,
    p12_f!, p13_f!, p14_f!, p15_f!, p16_f!, p17_f!, p18_f!, p19_f!, p20_f!, p21_f!,
    p22_f!, p23_f!)
dicts = (p1_dict, p2_dict, p3_dict, p4_dict, p5_dict, p6_dict, p7_dict, p8_dict, p9_dict,
    p10_dict, p11_dict, p12_dict, p13_dict, p14_dict, p15_dict, p16_dict, p17_dict,
    p18_dict, p19_dict, p20_dict, p21_dict, p22_dict, p23_dict)
algs = (NewtonRaphson(), TrustRegion(), LevenbergMarquardt())
names = ("NewtonRaphson", "TrustRegion", "LevenbergMarquardt")

for (problem, dict) in zip(problems, dicts)
    for (alg, name) in zip(algs, names)
        local x = dict["start"]
        local nlprob = NonlinearProblem(problem, x)
        local out = similar(x)
        try
            problem(out,
                solve(nlprob, alg, abstol = 1e-15, reltol = 1e-15).u, nothing)
            dict["error_" * name] = ""
        catch
            # println("Error in $name")
            dict["error_" * name] = "(Singular error)"
        end
        dict["out_" * name] = out
    end
    local x = dict["start"]
    local nlprob = NonlinearProblem(problem, x)
    sol = nlsolve(problem, x, xtol = 1e-15, ftol = 1e-15)
    dict["norm_nlsolve"] = sol.residual_norm
end

# ----------------------------------- Print results ----------------------------------------
i_str = i_str = rpad("nr", 3, " ")
title_str = rpad("Problem", 50, " ")
n_str = rpad("n", 5, " ")
norm_str = rpad(names[1], 20, " ") * rpad(names[2], 20, " ") * rpad(names[3], 20, " ") *
           rpad("nlsolve", 20, " ")
println("$i_str $title_str $n_str $norm_str")

for (i, dict) in enumerate(dicts)
    local i_str = rpad(string(i), 3, " ")
    local title_str = rpad(dict["title"], 50, " ")
    local n_str = rpad(string(dict["n"]), 5, " ")
    local norm_str = ""
    for (alg, name) in zip(algs, names)
        norm_str *= rpad(string(trunc(norm(dict["out_" * name]); sigdigits = 5)), 20, " ")
    end
    norm_str *= rpad(string(round(dict["norm_nlsolve"]; sigdigits = 5)), 20, " ")
    println("$i_str $title_str $n_str $norm_str")
end
