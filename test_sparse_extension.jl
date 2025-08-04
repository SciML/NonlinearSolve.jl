#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

println("=" ^ 50)
println("Testing NonlinearSolve without SparseArrays")
println("=" ^ 50)

# Test loading NonlinearSolve without SparseArrays
println("📦 Loading NonlinearSolve...")
load_time = @elapsed using NonlinearSolve
println("   ⏱️  Load time: $(round(load_time, digits=3))s")

# Test basic functionality
println("\n🧪 Testing basic NonlinearSolve functionality...")
try
    f(u, p) = u .* u .- p
    prob = NonlinearProblem(f, 0.1, 2.0)
    
    solve_time = @elapsed sol = solve(prob)
    println("   ✅ Basic solve works: u = $(sol.u)")
    println("   ⏱️  Solve time: $(round(solve_time, digits=3))s")
catch e
    println("   ❌ Basic solve failed: $e")
end

# Check memory usage
memory_mb = Sys.maxrss() / (1024^2)
println("\n💾 Memory usage: $(round(memory_mb, digits=2)) MB")

println("\n" * "=" ^ 50)
println("Testing with SparseArrays loaded")
println("=" ^ 50)

# Now test with SparseArrays loaded
println("📦 Loading SparseArrays...")
sparse_load_time = @elapsed using SparseArrays
println("   ⏱️  SparseArrays load time: $(round(sparse_load_time, digits=3))s")

# Test that sparse functionality works
println("\n🧪 Testing sparse functionality...")
try
    # Create a simple sparse matrix test
    using LinearAlgebra
    A = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0])
    println("   ✅ Can create sparse matrices: $(typeof(A))")
    
    # Test with a sparse Jacobian problem (simple example)
    function sparse_f!(du, u, p)
        du[1] = u[1]^2 - p
        du[2] = u[2]^2 - p
        du[3] = u[3]^2 - p
    end
    
    u0 = [0.1, 0.1, 0.1]
    prob_sparse = NonlinearProblem(sparse_f!, u0, 2.0)
    sol_sparse = solve(prob_sparse)
    println("   ✅ Sparse-compatible solve works: u = $(sol_sparse.u)")
    
catch e
    println("   ❌ Sparse functionality test failed: $e")
end

final_memory_mb = Sys.maxrss() / (1024^2)
memory_increase = final_memory_mb - memory_mb
println("\n💾 Memory after SparseArrays: $(round(final_memory_mb, digits=2)) MB")
println("   📈 Memory increase: $(round(memory_increase, digits=2)) MB")

println("\n✅ Extension test complete!")