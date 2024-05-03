# Using SimpleNonlinearSolve with KernelAbstractions.jl

We'll demonstrate how to leverage [SimpleNonlinearSolve.jl](https://github.com/SciML/SimpleNonlinearSolve.jl) inside kernels using [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl). This allows for efficient solving of very small nonlinear systems on GPUs by avoiding allocations and dynamic dispatch overhead. We'll use the generalized Rosenbrock problem as an example and solve it for multiple initial conditions on various GPU architectures.

### Prerequisites
Ensure the following packages are installed:
- Julia (v1.6 or later)
- NonlinearSolve.jl
- StaticArrays.jl
- KernelAbstractions.jl
- CUDA.jl (for NVIDIA GPUs)
- AMDGPU.jl (for AMD GPUs)

## Writing the Kernel
Define a kernel using **'@kernel'** from **'KernelAbstractions.jl'** to solve a single initial condition.

```@example kernel
using NonlinearSolve, StaticArrays
using KernelAbstractions, CUDA, AMDGPU

@kernel function parallel_nonlinearsolve_kernel!(result, @Const(prob), @Const(alg))
    i = @index(Global)
    prob_i = remake(prob; u0 = prob.u0[i])
    sol = solve(prob_i, alg)
    @inbounds result[i] = sol.u
    return nothing
end
```

## Vectorized Solving
Define a function to solve the problem for multiple initial conditions in parallel across GPU threads.

```@example kernel
function vectorized_solve(prob, alg; backend = CPU())
    result = KernelAbstractions.allocate(backend, eltype(prob.u0), length(prob.u0))
    groupsize = min(length(prob.u0), 1024)
    kernel! = parallel_nonlinearsolve_kernel!(backend, groupsize, length(prob.u0))
    kernel!(result, prob, alg)
    KernelAbstractions.synchronize(backend)
    return result
end
```

## Define the Rosenbrock Function
Define the generalized Rosenbrock function.

```@example kernel
@generated function generalized_rosenbrock(x::SVector{N}, p) where {N}
    vals = ntuple(i -> gensym(string(i)), N)
    expr = []
    push!(expr, :($(vals[1]) = oneunit(x[1]) - x[1]))
    for i in 2:N
        push!(expr, :($(vals[i]) = 10.0 * (x[$i] - x[$i - 1] * x[$i - 1])))
    end
    push!(expr, :(@SVector [$(vals...)]))
    return Expr(:block, expr...)
end
```

## Define the Problem
Create the nonlinear problem using the generalized Rosenbrock function and multiple initial conditions.

```@example kernel
u0 = @SVector [@SVector(rand(10)) for _ in 1:1024]
prob = NonlinearProblem(generalized_rosenbrock, u0)
```

## Solve the Problem
Solve the problem using **SimpleNonlinearSolve.jl** on different GPU architectures.

```@example kernel
# Threaded CPU
vectorized_solve(prob, SimpleNewtonRaphson(); backend = CPU())

# AMD ROCM GPU
vectorized_solve(prob, SimpleNewtonRaphson(); backend = ROCBackend())

# NVIDIA CUDA GPU
vectorized_solve(prob, SimpleNewtonRaphson(); backend = CUDABackend())
```

## Conclusion
This tutorial illustrated how to utilize **SimpleNonlinearSolve.jl** inside kernels using **KernelAbstractions.jl**, enabling efficient solving of small nonlinear systems on GPUs for applications requiring parallel processing and high performance.
