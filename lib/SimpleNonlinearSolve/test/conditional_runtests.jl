#!/usr/bin/env julia

"""
Conditional test runner for SimpleNonlinearSolve that properly handles Enzyme
dependencies based on Julia version to avoid precompilation failures.
"""

using Pkg

println("=" * 60)
println("SimpleNonlinearSolve Conditional Test Runner")
println("=" * 60)
println("Julia version: $(VERSION)")
println("VERSION.prerelease: $(VERSION.prerelease)")

# Check if Enzyme should be available
enzyme_available = false

if isempty(VERSION.prerelease)
    println("✅ Stable Julia version - attempting to load Enzyme")
    try
        # Try to add Enzyme if it's not already available
        try
            @eval using Enzyme
            enzyme_available = true
            println("✅ Enzyme is already available")
        catch
            println("⚠️  Enzyme not found, adding to environment...")
            Pkg.add(name="Enzyme", version="0.13.11")
            @eval using Enzyme
            enzyme_available = true  
            println("✅ Successfully added and loaded Enzyme")
        end
    catch e
        println("❌ Failed to load Enzyme: $e")
        println("   Tests will run without Enzyme support")
        enzyme_available = false
    end
else
    println("⚠️  Prerelease Julia version - skipping Enzyme")
    println("   This prevents compilation failures on prerelease versions")
    enzyme_available = false
end

# Set a global flag that tests can check
ENV["ENZYME_AVAILABLE"] = string(enzyme_available)

println()
println("Running tests with Enzyme support: $enzyme_available")
println("=" * 60)

# Run the actual tests
include("runtests.jl")