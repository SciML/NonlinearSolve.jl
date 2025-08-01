#!/usr/bin/env julia

"""
    setup_enzyme_for_testing()

Conditionally adds Enzyme to the current environment if running on a stable Julia version.
This prevents Enzyme precompilation failures on Julia prerelease versions.
"""
function setup_enzyme_for_testing()
    println("Julia version: $(VERSION)")
    println("VERSION.prerelease: $(VERSION.prerelease)")
    
    if isempty(VERSION.prerelease)
        println("✅ Stable Julia version detected - adding Enzyme to test environment")
        
        # Check if we're in a Pkg environment
        if !isfile("Project.toml") && !isfile("JuliaProject.toml")
            error("No Project.toml found in current directory")
        end
        
        # Add Enzyme using Pkg
        import Pkg
        
        # Check if Enzyme is already available
        try
            using Enzyme
            println("✅ Enzyme is already available")
            return true
        catch
            println("⚠️  Enzyme not found, adding to environment...")
            
            # Add Enzyme with specific version constraint if needed
            try
                Pkg.add(name="Enzyme", version="0.13.11")
                println("✅ Successfully added Enzyme")
                return true
            catch e
                println("❌ Failed to add Enzyme: $e")
                return false
            end
        end
    else
        println("⚠️  Prerelease Julia version detected - skipping Enzyme setup")
        println("   This is expected behavior to avoid Enzyme precompilation failures")
        return false
    end
end

# Run setup if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    setup_enzyme_for_testing()
end