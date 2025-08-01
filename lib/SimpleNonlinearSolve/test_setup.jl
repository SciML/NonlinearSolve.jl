#!/usr/bin/env julia

"""
Conditional test setup for SimpleNonlinearSolve that handles Enzyme dependencies
based on Julia version to avoid precompilation failures on prerelease versions.
"""

using Pkg

println("Setting up test environment for SimpleNonlinearSolve...")
println("Julia version: $(VERSION)")

# Always needed test dependencies
base_test_deps = [
    "Aqua",
    "DiffEqBase", 
    "ExplicitImports",
    "InteractiveUtils",
    "NonlinearProblemLibrary",
    "Pkg",
    "PolyesterForwardDiff",
    "Random",
    "ReverseDiff",
    "StaticArrays",
    "Test",
    "TestItemRunner",
    "Tracker",
    "Zygote"
]

# Add base dependencies
println("Adding base test dependencies...")
for dep in base_test_deps
    try
        Pkg.add(dep)
    catch e
        println("Warning: Failed to add $dep: $e")
    end
end

# Conditionally add Enzyme for stable versions only
if isempty(VERSION.prerelease)
    println("✅ Stable Julia version - adding Enzyme")
    try
        Pkg.add(name="Enzyme", version="0.13.11")
        println("✅ Enzyme added successfully")
    catch e
        println("❌ Failed to add Enzyme: $e")
    end
else
    println("⚠️  Prerelease Julia version - skipping Enzyme to avoid compilation failures")
end

println("Test environment setup complete!")