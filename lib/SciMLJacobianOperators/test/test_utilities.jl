# Test utilities for SciMLJacobianOperators

"""
    is_julia_prerelease()

Check if the current Julia version is a prerelease version (e.g., DEV, alpha, beta, rc).
"""
function is_julia_prerelease()
    version_string = string(VERSION)
    # Check for DEV versions (e.g., "1.12.0-DEV.1234")
    contains(version_string, "DEV") && return true
    # Check for alpha/beta/rc versions
    contains(version_string, "alpha") && return true
    contains(version_string, "beta") && return true
    contains(version_string, "rc") && return true
    return false
end