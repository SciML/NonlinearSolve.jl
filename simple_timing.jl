#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

println("=" ^ 80)
println("NonlinearSolve.jl Load Time Analysis")
println("=" ^ 80)

# Track compilation stages
deps_to_test = [
    "LinearAlgebra",
    "SparseArrays", 
    "ForwardDiff",
    "FiniteDiff",
    "LinearSolve",
    "SciMLBase",
    "DiffEqBase",
    "ArrayInterface",
    "PrecompileTools",
    "CommonSolve",
    "Reexport",
    "ConcreteStructs",
    "ADTypes",
    "FastClosures"
]

println("\n🔍 Testing individual dependency load times...")
dep_times = Dict{String, Float64}()

for dep in deps_to_test
    print("📦 Loading $dep... ")
    try
        time = @elapsed eval(Meta.parse("using $dep"))
        dep_times[dep] = time
        println("$(round(time, digits=3))s")
    catch e
        println("❌ FAILED: $e")
        dep_times[dep] = -1.0
    end
end

# Test subpackages
println("\n🏗️  Testing NonlinearSolve sub-packages...")
subpkg_times = Dict{String, Float64}()

subpackages = [
    "NonlinearSolveBase",
    "SimpleNonlinearSolve", 
    "BracketingNonlinearSolve",
    "NonlinearSolveFirstOrder",
    "NonlinearSolveQuasiNewton",
    "NonlinearSolveSpectralMethods"
]

for pkg in subpackages
    print("📦 Loading $pkg... ")
    try
        time = @elapsed eval(Meta.parse("using $pkg"))
        subpkg_times[pkg] = time
        println("$(round(time, digits=3))s")
    catch e
        println("❌ FAILED: $e")
        subpkg_times[pkg] = -1.0
    end
end

# Test main package
println("\n🚀 Loading main NonlinearSolve package...")
print("📦 Loading NonlinearSolve... ")
main_time = @elapsed using NonlinearSolve
println("$(round(main_time, digits=3))s")

# Memory analysis
println("\n💾 Memory usage analysis...")
memory_mb = Sys.maxrss() / (1024^2)
println("Current memory usage: $(round(memory_mb, digits=2)) MB")

# Generate report
println("\n" * "=" ^ 80)
println("📊 LOAD TIME REPORT")
println("=" ^ 80)

println("\n🏆 Top 5 slowest dependencies:")
sorted_deps = sort(collect(filter(p -> p[2] > 0, dep_times)), by=x->x[2], rev=true)
for (i, (name, time)) in enumerate(sorted_deps[1:min(5, length(sorted_deps))])
    println("$i. $(rpad(name, 20)) $(round(time, digits=3))s")
end

println("\n🏗️  Sub-package times:")
sorted_subpkgs = sort(collect(filter(p -> p[2] > 0, subpkg_times)), by=x->x[2], rev=true)
for (name, time) in sorted_subpkgs
    println("   $(rpad(name, 25)) $(round(time, digits=3))s")
end

total_deps = sum([x for x in values(dep_times) if x > 0])
total_subpkgs = sum([x for x in values(subpkg_times) if x > 0])

println("\n📈 SUMMARY:")
println("   Main NonlinearSolve load:    $(round(main_time, digits=3))s")
println("   Total dependencies:          $(round(total_deps, digits=3))s")
println("   Total sub-packages:          $(round(total_subpkgs, digits=3))s")
println("   Current memory usage:        $(round(memory_mb, digits=2)) MB")

# Find the biggest contributor
all_times = merge(dep_times, subpkg_times)
biggest = maximum([x for x in values(all_times) if x > 0])
biggest_component = ""
for (name, time) in all_times
    if time == biggest
        biggest_component = name
        break
    end
end

percentage = round(biggest / main_time * 100, digits=1)
println("\n🚨 BIGGEST LOAD TIME CONTRIBUTOR:")
println("   $biggest_component: $(round(biggest, digits=3))s ($percentage% of total)")

println("\n✅ Analysis complete!")