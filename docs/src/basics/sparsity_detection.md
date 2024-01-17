# [(Semi-)Automatic Sparsity Detection](@id sparsity-detection)

This section describes how to enable Sparsity Detection. For a detailed tutorial on how
to use this in an actual problem, see
[this tutorial on Efficiently Solving Large Sparse Ill-Conditioned Nonlinear Systems](@ref large_systems).

Notation wise we are trying to solve for `x` such that `nlfunc(x) = 0`.

## Case I: Sparse Jacobian Prototype is Provided

Let's say you have a Sparse Jacobian Prototype `jac_prototype`, in this case you can
create your problem as follows:

```julia
prob = NonlinearProblem(NonlinearFunction(nlfunc; sparsity = jac_prototype), x0)
# OR
prob = NonlinearProblem(NonlinearFunction(nlfunc; jac_prototype = jac_prototype), x0)
```

NonlinearSolve will automatically perform matrix coloring and use sparse differentiation.

Now you can help the solver further by providing the color vector. This can be done by

```julia
prob = NonlinearProblem(NonlinearFunction(nlfunc; sparsity = jac_prototype,
        colorvec = colorvec), x0)
# OR
prob = NonlinearProblem(NonlinearFunction(nlfunc; jac_prototype = jac_prototype,
        colorvec = colorvec), x0)
```

If the `colorvec` is not provided, then it is computed on demand.

!!! note
    
    One thing to be careful about in this case is that `colorvec` is dependent on the
    autodiff backend used. Forward Mode and Finite Differencing will assume that the
    colorvec is the column colorvec, while Reverse Mode will assume that the colorvec is the
    row colorvec.

## Case II: Sparsity Detection algorithm is provided

If you don't have a Sparse Jacobian Prototype, but you know the which sparsity detection
algorithm you want to use, then you can create your problem as follows:

```julia
prob = NonlinearProblem(NonlinearFunction(nlfunc; sparsity = SymbolicsSparsityDetection()),
    x0)  # Remember to have Symbolics.jl loaded
# OR
prob = NonlinearProblem(NonlinearFunction(nlfunc; sparsity = ApproximateJacobianSparsity()),
    x0)
```

These Detection Algorithms are from [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl),
refer to the documentation there for more details.

## Case III: Sparse AD Type is being Used

If you constructed a Nonlinear Solver with a sparse AD type, for example

```julia
NewtonRaphson(; autodiff = AutoSparseForwardDiff())
# OR
TrustRegion(; autodiff = AutoSparseZygote())
```

then NonlinearSolve will automatically perform matrix coloring and use sparse
differentiation if none of `sparsity` or `jac_prototype` is provided. If `Symbolics.jl` is
loaded, we default to using `SymbolicsSparsityDetection()`, else we default to using
`ApproximateJacobianSparsity()`.

`Case I/II` take precedence for sparsity detection and we perform sparse AD based on those
options if those are provided.

!!! warning
    
    If you provide a non-sparse AD, and provide a `sparsity` or `jac_prototype` then
    we will use dense AD. This is because, if you provide a specific AD type, we assume
    that you know what you are doing and want to override the default choice of `nothing`.
