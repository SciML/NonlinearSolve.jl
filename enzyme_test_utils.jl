"""
Universal Enzyme test utilities for NonlinearSolve.jl test suites.

This module provides utilities for conditionally loading and using Enzyme
in tests based on Julia version to prevent precompilation failures on
prerelease versions.
"""

"""
    setup_enzyme_for_testing()

Conditionally sets up Enzyme for testing based on Julia version.
Returns true if Enzyme is available for testing, false otherwise.

On stable Julia versions: Attempts to load Enzyme
On prerelease Julia versions: Skips Enzyme to prevent compilation failures
"""
function setup_enzyme_for_testing()
    enzyme_available = false
    
    if isempty(VERSION.prerelease)
        try
            @eval using Enzyme
            enzyme_available = true
        catch e
            # Enzyme not available - this is OK for some environments
            enzyme_available = false
        end
    else
        # Prerelease version - skip Enzyme to avoid compilation failures
        enzyme_available = false
    end
    
    return enzyme_available
end

"""
    add_enzyme_backends!(backends, enzyme_available::Bool)

Conditionally adds Enzyme autodiff backends to the provided array if Enzyme is available.
"""
function add_enzyme_backends!(backends, enzyme_available::Bool)
    if enzyme_available
        try
            push!(backends, AutoEnzyme())
        catch e
            @warn "Failed to add AutoEnzyme() backend: $e"
        end
    end
    return backends
end

"""
    add_enzyme_backends!(forward_ads, reverse_ads, enzyme_available::Bool)

Conditionally adds Enzyme autodiff backends to both forward and reverse AD arrays.
"""
function add_enzyme_backends!(forward_ads, reverse_ads, enzyme_available::Bool)
    if enzyme_available
        try
            # Add to reverse AD backends
            push!(reverse_ads, AutoEnzyme())
            push!(reverse_ads, AutoEnzyme(; mode = Enzyme.Reverse))
            
            # Add to forward AD backends  
            push!(forward_ads, AutoEnzyme())
            push!(forward_ads, AutoEnzyme(; mode = Enzyme.Forward))
        catch e
            @warn "Failed to add Enzyme backends: $e"
        end
    end
    return forward_ads, reverse_ads
end

# Export functions
export setup_enzyme_for_testing, add_enzyme_backends!