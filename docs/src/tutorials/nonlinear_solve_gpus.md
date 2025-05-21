# Accelerated Rootfinding on GPUs

NonlinearSolve.jl supports GPU acceleration on a wide array of devices, such as:

| GPU Manufacturer | GPU Kernel Language | Julia Support Package                              | Backend Type             |
|:---------------- |:------------------- |:-------------------------------------------------- |:------------------------ |
| NVIDIA           | CUDA                | [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)     | `CUDA.CUDABackend()`     |
| AMD              | ROCm                | [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) | `AMDGPU.ROCBackend()`    |
| Intel            | OneAPI              | [OneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) | `oneAPI.oneAPIBackend()` |
| Apple (M-Series) | Metal               | [Metal.jl](https://github.com/JuliaGPU/Metal.jl)   | `Metal.MetalBackend()`   |

To use NonlinearSolve.jl on GPUs, there are two distinctly different approaches:

 1. You can build a `NonlinearProblem` / `NonlinearLeastSquaresProblem` where the elements
    of the problem, i.e. `u0` and `p`, are defined on GPUs. This will make the evaluations
    of `f` occur on the GPU, and all internal updates of the solvers will be completely
    on the GPU as well. This is the optimal form for large systems of nonlinear equations.
 2. You can use SimpleNonlinearSolve.jl as kernels in KernelAbstractions.jl. This will build
    problem-specific GPU kernels in order to parallelize the solution of the chosen nonlinear
    system over a large number of inputs. This is useful for cases where you have a small
    `NonlinearProblem` / `NonlinearLeastSquaresProblem` which you want to solve over a large
    number of initial guesses or parameters.

For a deeper dive into the computational difference between these techniques and why it
leads to different pros/cons, see the
[DiffEqGPU.jl technical paper](https://www.sciencedirect.com/science/article/abs/pii/S0045782523007156).
In particular, the second form is unique to NonlinearSolve.jl and offers orders of magnitude
performance improvements over libraries in Jax and PyTorch which are restricted to only
using the first form.

In this tutorial we will highlight both use cases in separate parts.

!!! note

    If you're looking for GPU-accelerated neural networks inside of nonlinear solvers,
    check out [DeepEquilibriumNetworks.jl](https://docs.sciml.ai/DeepEquilibriumNetworks/stable/).

## GPU Acceleration of Large Nonlinear Systems using GPUArrays

The simplest way to GPU accelerate a large system is to simply make your `u0` and `p` values
be on the GPU via GPUArrays. For example, CUDA.jl has the CuArray type which implements
standard array operations, such as broadcasting and linear algebra. And since
NonlinearSolve.jl respects your chosen array types, if you choose to make `u0` be a type
that is on the GPU, then all internal broadcasting and linear algebra takes place on the
GPU.

This means the one major limitation is that you as a user must write your `f` to be
compatible with GPU arrays. Those limitations are discussed in detail in the GPU libraries,
for example
[the CuArray documentation discusses the operations available for CUDA arrays](https://cuda.juliagpu.org/stable/usage/array/)

In practice when this comes together, it looks like:

```julia
using NonlinearSolve, CUDA

f(u, p) = u .* u .- p
u0 = cu(ones(1000))
p = cu(collect(1:1000))
prob = NonlinearProblem(f, u0, p)
sol = solve(prob, NewtonRaphson(), abstol=1f-4)
```

Notice a few things here. One, nothing is different except the input array types. But
notice that `cu` arrays automatically default to `Float32` precision. Since NonlinearSolve.jl
respects the user's chosen types, this changes NonlinearSolve.jl to use `Float32` precision,
and thus the tolerances are adjusted accordingly.

## GPU Acceleration over Large Parameter Searches using KernelAbstractions.jl

If one has a "small" (200 equations or less) system of equations which they wish to solve
over many different inputs (parameters), then using the kernel generation approach will be
much more efficient than using the GPU-based array approach. In short, the GPU array
approach strings together standard GPU kernel calls (matrix multiply, +, etc.) where each
operation is an optimized GPU-accelerated call. In the kernel-building approach, we build
a custom kernel `f` that is then compiled specifically for our problem and ran in parallel.
This is equivalent to having built custom CUDA code for our problem! The reason this is
so much faster is because each kernel call has startup overhead, and we can cut that all
down to simply one optimized call.

To do this, we use KernelAbstractions.jl. First we have to say "what" our kernel is. The
kernel is the thing you want to embaressingly parallel call a bunch. For this nonlinear
solving, it will be the rebuilding of our nonlinear problem to new parameters and solving
it. This function must be defined using the `KernelAbstractions.@kernel` macro. This looks
like:

```julia
using NonlinearSolve, StaticArrays
using KernelAbstractions # For writing vendor-agnostic kernels
using CUDA # For if you have an NVIDIA GPU
using AMDGPU # For if you have an AMD GPU
using Metal # For if you have a Mac M-series device and want to use the built-in GPU
using OneAPI # For if you have an Intel GPU

@kernel function parallel_nonlinearsolve_kernel!(result, @Const(prob), @Const(alg))
    i = @index(Global)
    prob_i = remake(prob; p = prob.p[i])
    sol = solve(prob_i, alg)
    @inbounds result[i] = sol.u
end
```

Note that `i = @index(Global)` is used to get a global index. I.e. this kernel will be
called with `N` different `prob` objects, and this `i` means "for the ith call". So this
is saying, "for the ith call, get the i'th parameter set and solve with these parameters.
The ith result is then this solution".

!!! note

    Because kernel code needs to be able to be compiled to a GPU kernel, it has very strict
    specifications of what's allowed because GPU cores are not as flexible as CPU cores.
    In general, this means that you need to avoid any runtime operations in kernel code,
    such as allocating vectors, dynamic dispatch, type instabilities, etc. The main thing
    to note is that most NonlinearSolve.jl algorithms will not be compatible with being
    in kernels. However, **SimpleNonlinearSolve.jl solvers are tested to be compatible**,
    and thus one should only choose SimpleNonlinearSolve.jl methods within kernels.

Once you have defined your kernel, you need to use KernelAbstractions in order to distribute
your call. This looks like:

```julia
function vectorized_solve(prob, alg; backend = CPU())
    result = KernelAbstractions.allocate(backend, eltype(prob.p), length(prob.p))
    groupsize = min(length(prob.p), 1024)
    kernel! = parallel_nonlinearsolve_kernel!(backend, groupsize, length(prob.p))
    kernel!(result, prob, alg)
    KernelAbstractions.synchronize(backend)
    return result
end
```

Now let's build a nonlinear system to test it on.

```julia
@inbounds function p2_f(x, p)
    out1 = x[1] + p[1] * x[2]
    out2 = sqrt(p[2]) * (x[3] - x[4])
    out3 = (x[2] - p[3] * x[3])^2
    out4 = sqrt(p[4]) * (x[1] - x[4]) * (x[1] - x[4])
    SA[out1,out2,out3,out4]
end

p = @SVector [@SVector(rand(Float32, 4)) for _ in 1:1024]
u0 = SA[1f0, 2f0, 3f0, 4f0]
prob = SciMLBase.ImmutableNonlinearProblem{false}(p2_f, u0, p)
```

!!! note

    Because the custom kernel is going to need to embed the the code for our nonlinear
    problem into the kernel, it also must be written to be GPU compatible.
    In general, this means that you need to avoid any runtime operations in kernel code,
    such as allocating vectors, dynamic dispatch, type instabilities, etc. Thus to make this
    work, your `f` function should be non-allocating, your `u0` function should use
    StaticArrays, and you must use `SciMLBase.ImmutableNonlinearProblem`
    (which is exactly the same as NonlinearProblem except it's immutable to satisfy the
    requirements of GPU kernels). Also, it's recommended that for most GPUs you use Float32
    precision because many GPUs are much slower on 64-bit floating point operations.

and we then simply call our vectorized kernel to parallelize it:

```julia
# Threaded CPU
vectorized_solve(prob, SimpleNewtonRaphson(); backend = CPU())
# AMD ROCM GPU
vectorized_solve(prob, SimpleNewtonRaphson(); backend = ROCBackend())
# NVIDIA CUDA GPU
vectorized_solve(prob, SimpleNewtonRaphson(); backend = CUDABackend())
# Intel GPU
vectorized_solve(prob, SimpleNewtonRaphson(); backend = oneAPI.oneAPIBackend())
# Mac M-Series, such as M3Max
vectorized_solve(prob, SimpleNewtonRaphson(); backend = Metal.MetalBackend())
```

!!! warn
    The GPU-based calls will only work on your machine if you have a compatible GPU!
