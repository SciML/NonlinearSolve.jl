#!/usr/bin/env julia

"""
Universal Enzyme setup script for NonlinearSolve.jl test suites.

This script conditionally adds Enzyme to the test environment based on the Julia version.
It prevents Enzyme precompilation failures on Julia prerelease versions while ensuring
Enzyme tests run on stable versions.

Usage:
    julia test_enzyme_setup.jl

Environment Variables Set:
    ENZYME_AVAILABLE: "true" if Enzyme is available, "false" otherwise
"""

using Pkg

function setup_enzyme_environment()
    println("=" * 70)
    println("NonlinearSolve.jl Enzyme Test Environment Setup")
    println("=" * 70)
    println("Julia version: $(VERSION)")
    println("Prerelease components: $(VERSION.prerelease)")
    
    enzyme_available = false
    
    if isempty(VERSION.prerelease)
        println("✅ Running on stable Julia version")
        println("   Attempting to set up Enzyme for testing...")
        
        try
            # First check if Enzyme is already available
            try
                @eval using Enzyme
                println("✅ Enzyme is already loaded")
                enzyme_available = true
            catch LoadError
                println("⚠️  Enzyme not found in environment, attempting to add...")
                
                # Add Enzyme with version constraint to avoid conflicts
                Pkg.add(name="Enzyme", version="0.13.11")
                @eval using Enzyme
                println("✅ Successfully added and loaded Enzyme")
                enzyme_available = true
            end
            
        catch e
            println("❌ Failed to set up Enzyme: $(typeof(e)): $e")
            println("   Tests will proceed without Enzyme support")
            enzyme_available = false
        end
        
    else
        println("⚠️  Running on Julia prerelease version")
        println("   Skipping Enzyme setup to prevent compilation failures")
        println("   This is expected behavior for prerelease compatibility")
        enzyme_available = false
    end
    
    # Set environment variable for test files to check
    ENV["ENZYME_AVAILABLE"] = string(enzyme_available)
    
    println()
    println("Setup Summary:")
    println("  Enzyme available for tests: $enzyme_available")
    println("  Environment variable set: ENZYME_AVAILABLE=$(ENV["ENZYME_AVAILABLE"])")
    println("=" * 70)
    
    return enzyme_available
end

# Run setup if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    setup_enzyme_environment()
end