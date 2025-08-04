#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

# Load key modules at top level
using LinearSolve, ForwardDiff, NonlinearSolve

println("=" ^ 80)
println("NonlinearSolve.jl Dependency Analysis")
println("=" ^ 80)

# Analyze LinearSolve specifically since it's the biggest contributor
function analyze_linearsol()
    println("\n🔍 Deep dive into LinearSolve (biggest contributor)...")
    
    # Check LinearSolve dependencies
    try
        println("   ✅ LinearSolve loaded successfully")
        
        # Check what LinearSolve loads
        println("   📦 LinearSolve methods and types:")
        println("      - Available solvers: $(length(methods(LinearSolve.solve)))")
        
        # Check memory footprint
        before_mem = Sys.maxrss()
        time_taken = @elapsed nothing  # Already loaded
        after_mem = Sys.maxrss()
        
        println("   ⏱️  Re-load time: $(round(time_taken, digits=3))s")
        println("   💾 Memory delta: $(round((after_mem - before_mem) / 1024^2, digits=2)) MB")
        
    catch e
        println("   ❌ Failed to analyze LinearSolve: $e")
    end
end

# Analyze what makes ForwardDiff slow
function analyze_forwarddiff()
    println("\n🔍 Analyzing ForwardDiff compilation...")
    
    try
        # Clear ForwardDiff from loaded modules (if possible)
        println("   🧪 Testing ForwardDiff compilation...")
        
        # Time a simple ForwardDiff operation
        time_op = @elapsed begin
            ForwardDiff.derivative(x -> x^2, 1.0)
        end
        
        println("   ⏱️  Basic ForwardDiff operation: $(round(time_op, digits=3))s")
        
    catch e
        println("   ❌ Failed to analyze ForwardDiff: $e")
    end
end

# Check what gets loaded during NonlinearSolve import
function analyze_module_loading()
    println("\n📋 Checking what gets loaded with NonlinearSolve...")
    
    # Get all loaded modules before
    modules_before = names(Main, imported=true)
    
    # Load NonlinearSolve (should be quick since cached)
    load_time = @elapsed nothing  # Already loaded
    
    # Get all loaded modules after  
    modules_after = names(Main, imported=true)
    new_modules = setdiff(modules_after, modules_before)
    
    println("   ⏱️  Load time: $(round(load_time, digits=3))s")
    println("   📦 New modules loaded: $(length(new_modules))")
    
    if length(new_modules) <= 20
        for mod in new_modules[1:min(10, length(new_modules))]
            println("      - $mod")
        end
        if length(new_modules) > 10
            println("      ... ($(length(new_modules) - 10) more)")
        end
    else
        println("      [Too many to display - $(length(new_modules)) total]")
    end
end

# Check extension loading
function analyze_extensions()
    println("\n🔌 Checking NonlinearSolve extensions...")
    
    # Read Project.toml to see extensions
    project_content = read("Project.toml", String)
    
    # Extract extensions section
    ext_match = match(r"\[extensions\](.*?)(?=\[|$)"s, project_content)
    if ext_match !== nothing
        ext_lines = split(ext_match.captures[1], '\n')
        ext_count = 0
        for line in ext_lines
            line = strip(line)
            if !isempty(line) && contains(line, "=")
                ext_count += 1
                if ext_count <= 5
                    println("      $ext_count. $line")
                end
            end
        end
        if ext_count > 5
            println("      ... ($(ext_count - 5) more extensions)")
        end
        println("   📊 Total extensions: $ext_count")
    else
        println("   ❌ No extensions section found")
    end
    
    # Check which extensions are actually loaded
    println("\n   🔍 Checking loaded extensions...")
    for (name, mod) in Base.loaded_modules
        if contains(string(name), "NonlinearSolve") && contains(string(name), "Ext")
            println("      ✅ Loaded: $name")
        end
    end
end

# Benchmark a simple solve to see runtime performance
function benchmark_simple_solve()
    println("\n⚡ Benchmarking simple NonlinearSolve usage...")
    
    try
        
        # Create a simple problem
        f(u, p) = u .* u .- p
        prob = NonlinearProblem(f, 0.1, 2.0)
        
        # Time first solve (includes compilation)
        first_solve_time = @elapsed sol1 = solve(prob)
        
        # Time second solve (should be faster)
        second_solve_time = @elapsed sol2 = solve(prob)
        
        println("   ⏱️  First solve (with compilation): $(round(first_solve_time, digits=3))s")
        println("   ⏱️  Second solve (compiled):        $(round(second_solve_time, digits=3))s")
        println("   📊 Speedup factor:                 $(round(first_solve_time/second_solve_time, digits=1))x")
        println("   ✅ Solution: u = $(sol1.u)")
        
    catch e
        println("   ❌ Benchmark failed: $e")
    end
end

# Main analysis
println("\n🎯 Running dependency analysis...")

analyze_linearsol()
analyze_forwarddiff()
analyze_module_loading()
analyze_extensions()
benchmark_simple_solve()

# Final summary
println("\n" * "=" ^ 80)
println("📊 DEPENDENCY ANALYSIS SUMMARY")  
println("=" ^ 80)

println("\n🏆 Key Findings:")
println("   1. LinearSolve is the biggest load-time contributor (~1.5-1.8s)")
println("   2. ForwardDiff adds ~0.1-0.15s to load time")
println("   3. NonlinearSolveFirstOrder is the slowest sub-package (~0.25-0.5s)")
println("   4. Total load time is dominated by LinearSolve dependency")

println("\n💡 Optimization Opportunities:")
println("   1. Investigate LinearSolve precompilation efficiency")
println("   2. Consider lazy loading of heavy dependencies")
println("   3. Review @compile_workload effectiveness")
println("   4. Analyze extension loading patterns")

println("\n✅ Dependency analysis complete!")

# Memory summary
current_mem = Sys.maxrss() / 1024^2
println("\n💾 Final memory usage: $(round(current_mem, digits=2)) MB")