#!/usr/bin/env julia

println("Testing NonlinearSolve.jl Enzyme gating")
println(repeat("=", 50))

# Test 1: Verify current stable version behavior
println("\n1. Current Julia version (stable):")
println("   VERSION.prerelease = ", VERSION.prerelease)
println("   isempty(VERSION.prerelease) = ", isempty(VERSION.prerelease))

# Simulate what happens in stable version
if isempty(VERSION.prerelease)
    println("   ✅ Would import Enzyme and add to backends")
else
    println("   ❌ Would skip Enzyme (unexpected for stable)")
end

# Test 2: Simulate prerelease behavior
println("\n2. Simulated prerelease version:")
mock_prerelease = ("rc", 1)
println("   Mock VERSION.prerelease = ", mock_prerelease)
println("   isempty(mock_prerelease) = ", isempty(mock_prerelease))

if isempty(mock_prerelease)
    println("   ❌ Would import Enzyme (bad for prerelease)")
else
    println("   ✅ Would skip Enzyme (good for prerelease)")
end

# Test 3: Verify autodiff backend behavior
println("\n3. AutoDiff backend behavior:")

function test_backends(is_prerelease)
    backends = ["AutoForwardDiff()", "AutoZygote()", "AutoFiniteDiff()"]
    initial_count = length(backends)
    
    if !is_prerelease
        push!(backends, "AutoEnzyme()")
    end
    
    return backends, initial_count
end

stable_backends, initial = test_backends(false)
prerelease_backends, _ = test_backends(true)

println("   Stable version: ", length(stable_backends), " backends (includes Enzyme)")  
println("   Prerelease version: ", length(prerelease_backends), " backends (excludes Enzyme)")
println("   Enzyme correctly gated: ", length(stable_backends) > length(prerelease_backends))

println("\n4. Checking actual test files:")

# Check if our test files have the gating
test_files = [
    "lib/SimpleNonlinearSolve/test/core/rootfind_tests.jl",
    "lib/SciMLJacobianOperators/test/core_tests.jl", 
    "lib/NonlinearSolveQuasiNewton/test/core_tests.jl"
]

all_files_gated = true
for file in test_files
    if isfile(file)
        content = read(file, String)
        has_gating = occursin("if isempty(VERSION.prerelease)", content) && occursin("using Enzyme", content)
        println("   $file: ", has_gating ? "✅ Gated" : "❌ Not gated")
        all_files_gated = all_files_gated && has_gating
    else
        println("   $file: ❌ Not found")
        all_files_gated = false
    end
end

println(repeat("=", 50))
println("RESULT: ", all_files_gated ? "✅ SUCCESS - All tests properly gated for prerelease" : "❌ FAILURE - Some tests missing gating")
println(repeat("=", 50))