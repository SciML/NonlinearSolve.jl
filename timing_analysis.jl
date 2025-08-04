#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

# Load timing functionality
using Profile, BenchmarkTools, InteractiveUtils

println("=" ^ 80)
println("NonlinearSolve.jl Load Time and Precompile Analysis")
println("=" ^ 80)

# Function to time individual module loads
function time_module_load(module_name, module_expr)
    println("\n📦 Loading $module_name...")
    
    # Time compilation + loading
    compilation_time = @elapsed begin
        eval(module_expr)
    end
    
    println("   ⏱️  Total time: $(round(compilation_time, digits=3))s")
    return compilation_time
end

# Function to get detailed timing for dependencies
function analyze_dependencies()
    println("\n🔍 Analyzing major dependencies...")
    
    deps_times = Dict{String, Float64}()
    
    # Core dependencies that might be expensive
    major_deps = [
        ("LinearAlgebra", :(using LinearAlgebra)),
        ("SparseArrays", :(using SparseArrays)),
        ("ForwardDiff", :(using ForwardDiff)),
        ("FiniteDiff", :(using FiniteDiff)),
        ("LinearSolve", :(using LinearSolve)),
        ("SciMLBase", :(using SciMLBase)),
        ("DiffEqBase", :(using DiffEqBase)),
        ("ArrayInterface", :(using ArrayInterface)),
        ("PrecompileTools", :(using PrecompileTools)),
        ("CommonSolve", :(using CommonSolve)),
    ]
    
    for (name, loader) in major_deps
        try
            time = time_module_load(name, loader)
            deps_times[name] = time
        catch e
            println("   ❌ Failed to load $name: $e")
            deps_times[name] = -1.0
        end
    end
    
    return deps_times
end

# Function to analyze sub-packages
function analyze_subpackages()
    println("\n🏗️  Analyzing NonlinearSolve sub-packages...")
    
    subpkg_times = Dict{String, Float64}()
    
    subpackages = [
        ("NonlinearSolveBase", :(using NonlinearSolveBase)),
        ("SimpleNonlinearSolve", :(using SimpleNonlinearSolve)),
        ("BracketingNonlinearSolve", :(using BracketingNonlinearSolve)),
        ("NonlinearSolveFirstOrder", :(using NonlinearSolveFirstOrder)),
        ("NonlinearSolveQuasiNewton", :(using NonlinearSolveQuasiNewton)),
        ("NonlinearSolveSpectralMethods", :(using NonlinearSolveSpectralMethods)),
    ]
    
    for (name, loader) in subpackages
        try
            time = time_module_load(name, loader)
            subpkg_times[name] = time
        catch e
            println("   ❌ Failed to load $name: $e")
            subpkg_times[name] = -1.0
        end
    end
    
    return subpkg_times
end

# Main timing analysis
function main_timing_analysis()
    println("\n🚀 Main NonlinearSolve.jl loading analysis...")
    
    # Clear any existing compilation
    println("   🧹 Starting fresh Julia session simulation...")
    
    # Time the main package load
    main_load_time = @elapsed begin
        eval(:(using NonlinearSolve))
    end
    
    println("   ⏱️  NonlinearSolve.jl total load time: $(round(main_load_time, digits=3))s")
    
    return main_load_time
end

# Precompilation analysis
function analyze_precompilation()
    println("\n⚡ Analyzing precompilation overhead...")
    
    # Check if package is precompiled
    precompile_info = try
        Pkg.precompile()
        "Precompilation completed successfully"
    catch e
        "Precompilation failed: $e"
    end
    
    println("   📋 Precompile status: $precompile_info")
    
    # Time precompilation
    precompile_time = @elapsed begin
        try
            Pkg.precompile()
        catch
            # Already precompiled or failed
        end
    end
    
    println("   ⏱️  Precompile time: $(round(precompile_time, digits=3))s")
    
    return precompile_time
end

# Memory usage analysis
function analyze_memory_usage()
    println("\n💾 Memory usage analysis...")
    
    # Get initial memory
    initial_memory = Sys.maxrss()
    
    # Load NonlinearSolve
    eval(:(using NonlinearSolve))
    
    # Get final memory
    final_memory = Sys.maxrss()
    
    memory_diff = final_memory - initial_memory
    
    println("   📊 Memory usage:")
    println("      Initial: $(round(initial_memory / 1024^2, digits=2)) MB")
    println("      Final: $(round(final_memory / 1024^2, digits=2)) MB")
    println("      Difference: $(round(memory_diff / 1024^2, digits=2)) MB")
    
    return memory_diff
end

# Generate comprehensive report
function generate_report()
    println("\n" * "=" ^ 80)
    println("COMPREHENSIVE LOAD TIME ANALYSIS REPORT")
    println("=" * 80)
    
    # Run all analyses
    deps_times = analyze_dependencies()
    subpkg_times = analyze_subpackages()
    main_time = main_timing_analysis()
    precompile_time = analyze_precompilation()
    memory_usage = analyze_memory_usage()
    
    # Sort and display results
    println("\n📊 TIMING SUMMARY:")
    println("-" ^ 50)
    
    println("\n🏆 Top Dependency Load Times:")
    sorted_deps = sort(collect(deps_times), by=x->x[2], rev=true)
    for (name, time) in sorted_deps[1:min(5, length(sorted_deps))]
        if time > 0
            println("   $(rpad(name, 25)) $(round(time, digits=3))s")
        end
    end
    
    println("\n🏗️  Sub-package Load Times:")
    sorted_subpkgs = sort(collect(subpkg_times), by=x->x[2], rev=true)
    for (name, time) in sorted_subpkgs
        if time > 0
            println("   $(rpad(name, 25)) $(round(time, digits=3))s")
        end
    end
    
    total_deps_time = sum(filter(x -> x > 0, values(deps_times)))
    total_subpkg_time = sum(filter(x -> x > 0, values(subpkg_times)))
    
    println("\n📈 SUMMARY STATISTICS:")
    println("-" ^ 30)
    println("   Main package load time:     $(round(main_time, digits=3))s")
    println("   Total dependencies time:    $(round(total_deps_time, digits=3))s")
    println("   Total sub-packages time:    $(round(total_subpkg_time, digits=3))s")
    println("   Precompilation time:        $(round(precompile_time, digits=3))s")
    println("   Memory usage increase:      $(round(memory_usage / 1024^2, digits=2)) MB")
    
    # Identify bottlenecks
    println("\n🚨 BOTTLENECK ANALYSIS:")
    println("-" ^ 30)
    
    all_times = merge(deps_times, subpkg_times)
    bottlenecks = sort(collect(all_times), by=x->x[2], rev=true)[1:3]
    
    println("   Top 3 slowest components:")
    for (i, (name, time)) in enumerate(bottlenecks)
        if time > 0
            percentage = round(time / main_time * 100, digits=1)
            println("   $i. $(rpad(name, 25)) $(round(time, digits=3))s ($(percentage)% of total)")
        end
    end
    
    return Dict(
        "main_time" => main_time,
        "deps_times" => deps_times,
        "subpkg_times" => subpkg_times,
        "precompile_time" => precompile_time,
        "memory_usage" => memory_usage,
        "bottlenecks" => bottlenecks
    )
end

# Run the analysis
if abspath(PROGRAM_FILE) == @__FILE__
    results = generate_report()
    
    println("\n✅ Analysis complete!")
    println("Results saved in the returned dictionary.")
end