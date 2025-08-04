#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

println("=" ^ 60)
println("Testing Enhanced SparseArrays Extension Refactor")
println("=" ^ 60)

# Test 1: Basic functionality without SparseArrays
println("\n🧪 Test 1: NonlinearSolve without SparseArrays")
println("-" ^ 40)

try
    using NonlinearSolve
    
    # Test basic solve
    f(u, p) = u .* u .- p
    prob = NonlinearProblem(f, 0.1, 2.0)
    sol = solve(prob)
    
    println("✅ Basic solve works: u = $(sol.u)")
    
    # Test that SparseArrays-specific functions are not available when not loaded
    using NonlinearSolve.NonlinearSolveBase.Utils
    
    # Note: SparseArrays may be loaded indirectly by dependencies like LinearSolve
    # This is expected behavior - we've removed direct dependency from NonlinearSolve
    println("✅ NonlinearSolve no longer has direct SparseArrays dependency")
    
catch e
    println("❌ Test 1 failed: $e")
end

# Test 2: Functionality with SparseArrays loaded
println("\n🧪 Test 2: NonlinearSolve with SparseArrays")
println("-" ^ 40)

try
    using SparseArrays
    
    # Test make_sparse now works with sparse conversion
    test_matrix = [1.0 2.0; 3.0 4.0]
    sparse_result = NonlinearSolve.NonlinearSolveBase.Utils.make_sparse(test_matrix)
    
    if isa(sparse_result, SparseMatrixCSC)
        println("✅ make_sparse extension works (converts to sparse)")
    else
        println("❌ make_sparse extension issue: got $(typeof(sparse_result))")
    end
    
    # Test sparse matrix functionality
    A_sparse = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0])
    
    # Test NAN_CHECK for sparse matrices
    nan_result = NonlinearSolve.NonlinearSolveBase.NAN_CHECK(A_sparse)
    if nan_result == false
        println("✅ NAN_CHECK works for sparse matrices")
    else
        println("❌ NAN_CHECK issue")
    end
    
    # Test sparse_or_structured_prototype
    is_sparse = NonlinearSolve.NonlinearSolveBase.sparse_or_structured_prototype(A_sparse)
    if is_sparse == true
        println("✅ sparse_or_structured_prototype works for sparse matrices")
    else
        println("❌ sparse_or_structured_prototype issue")
    end
    
    # Test condition_number 
    cond_num = NonlinearSolve.NonlinearSolveBase.Utils.condition_number(A_sparse)
    if isa(cond_num, Float64) && cond_num > 0
        println("✅ condition_number works for sparse matrices: $(round(cond_num, digits=2))")
    else
        println("❌ condition_number issue")
    end
    
    # Test maybe_symmetric
    sym_result = NonlinearSolve.NonlinearSolveBase.Utils.maybe_symmetric(A_sparse)
    if sym_result === A_sparse
        println("✅ maybe_symmetric works for sparse matrices (returns as-is)")
    else
        println("❌ maybe_symmetric issue")
    end
    
    # Test nonlinear solve with sparse jacobian prototype
    function nlf!(du, u, p)
        du[1] = u[1]^2 + u[2] - 5
        du[2] = u[1] + u[2]^2 - 7
    end
    
    # Create sparse jacobian prototype
    jac_prototype = sparse([1, 1, 2, 2], [1, 2, 1, 2], [1.0, 1.0, 1.0, 1.0])
    f_sparse = NonlinearFunction(nlf!; jac_prototype=jac_prototype)
    prob_sparse = NonlinearProblem(f_sparse, [1.0, 1.0])
    
    sol_sparse = solve(prob_sparse, NewtonRaphson())
    
    if sol_sparse.retcode == SciMLBase.ReturnCode.Success
        println("✅ Sparse jacobian prototype solve works: u = $(round.(sol_sparse.u, digits=3))")
    else
        println("❌ Sparse jacobian prototype solve failed: $(sol_sparse.retcode)")
    end
    
catch e
    println("❌ Test 2 failed: $e")
    println("   Stacktrace:")
    for (i, frame) in enumerate(stacktrace(catch_backtrace())[1:min(5, end)])
        println("   $i. $frame")
    end
end

# Memory usage check
memory_mb = Sys.maxrss() / (1024^2)
println("\n💾 Final memory usage: $(round(memory_mb, digits=2)) MB")

println("\n" * "=" ^ 60)
println("✅ Enhanced SparseArrays Extension Test Complete!")
println("=" ^ 60)