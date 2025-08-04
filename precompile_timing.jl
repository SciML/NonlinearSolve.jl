#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

println("=" ^ 80)
println("NonlinearSolve.jl Precompile Time Analysis")
println("=" ^ 80)

# Function to time precompilation of individual packages
function time_precompile_package(pkg_name)
    println("\n⚡ Timing precompilation of $pkg_name...")
    
    # Remove existing compiled cache
    depot_path = first(DEPOT_PATH)
    compiled_path = joinpath(depot_path, "compiled", "v$(VERSION.major).$(VERSION.minor)")
    
    # Find and remove compiled files for this package
    try
        for (root, dirs, files) in walkdir(compiled_path)
            for file in files
                if startswith(file, pkg_name)
                    rm(joinpath(root, file), force=true)
                    println("   🗑️  Removed cache: $file")
                end
            end
        end
    catch e
        println("   ⚠️  Cache cleanup failed: $e")
    end
    
    # Time the precompilation
    compile_time = @elapsed begin
        try
            eval(Meta.parse("using $pkg_name"))
        catch e
            println("   ❌ Failed to compile $pkg_name: $e")
            return -1.0
        end
    end
    
    println("   ⏱️  Precompile time: $(round(compile_time, digits=3))s")
    return compile_time
end

# Analyze precompile workload from source
function analyze_precompile_workload()
    println("\n🔬 Analyzing @compile_workload in NonlinearSolve.jl...")
    
    # Read the main source file
    src_file = "src/NonlinearSolve.jl"
    content = read(src_file, String)
    
    # Look for @setup_workload and @compile_workload blocks
    setup_match = match(r"@setup_workload begin(.*?)end"s, content)
    compile_match = match(r"@compile_workload begin(.*?)end"s, content)
    
    if setup_match !== nothing
        println("   📋 Found @setup_workload block:")
        setup_lines = split(setup_match.captures[1], '\n')
        for (i, line) in enumerate(setup_lines[1:min(10, length(setup_lines))])
            line = strip(line)
            if !isempty(line)
                println("      $i. $line")
            end
        end
        if length(setup_lines) > 10
            println("      ... ($(length(setup_lines) - 10) more lines)")
        end
    end
    
    if compile_match !== nothing
        println("   📋 Found @compile_workload block:")
        compile_lines = split(compile_match.captures[1], '\n')
        for (i, line) in enumerate(compile_lines[1:min(10, length(compile_lines))])
            line = strip(line)
            if !isempty(line)
                println("      $i. $line")
            end
        end
        if length(compile_lines) > 10
            println("      ... ($(length(compile_lines) - 10) more lines)")
        end
    end
    
    return setup_match !== nothing, compile_match !== nothing
end

# Time fresh precompilation
function time_fresh_precompilation()
    println("\n🔄 Timing fresh precompilation from scratch...")
    
    # Clear all compiled cache
    try
        depot_path = first(DEPOT_PATH)
        compiled_path = joinpath(depot_path, "compiled", "v$(VERSION.major).$(VERSION.minor)")
        
        println("   🧹 Clearing compiled cache at: $compiled_path")
        if isdir(compiled_path)
            rm(compiled_path, recursive=true, force=true)
            mkdir(compiled_path)
        end
    catch e
        println("   ⚠️  Cache clear failed: $e")
    end
    
    # Time full precompilation
    println("   ⏳ Running fresh precompilation...")
    precompile_time = @elapsed begin
        try
            Pkg.precompile()
        catch e
            println("   ❌ Precompilation failed: $e")
            return -1.0
        end
    end
    
    println("   ⏱️  Total precompile time: $(round(precompile_time, digits=3))s")
    return precompile_time
end

# Run analysis
println("\n🎯 Starting precompile analysis...")

# Analyze workload
has_setup, has_compile = analyze_precompile_workload()

# Test individual package timing (already compiled)
println("\n📦 Individual package load times (from cache):")
packages = [
    "LinearSolve",
    "ForwardDiff", 
    "NonlinearSolveFirstOrder",
    "SimpleNonlinearSolve",
    "NonlinearSolveQuasiNewton"
]

cached_times = Dict{String, Float64}()
for pkg in packages
    time = @elapsed eval(Meta.parse("using $pkg"))
    cached_times[pkg] = time
    println("   $(rpad(pkg, 25)) $(round(time, digits=3))s")
end

# Time main package load
println("\n🚀 Main package load (from cache):")
main_cached_time = @elapsed using NonlinearSolve
println("   NonlinearSolve:              $(round(main_cached_time, digits=3))s")

# Generate report
println("\n" * "=" ^ 80)
println("📊 PRECOMPILE ANALYSIS REPORT")
println("=" ^ 80)

println("\n📋 Precompile Workload:")
println("   Has @setup_workload:        $has_setup")
println("   Has @compile_workload:      $has_compile")

println("\n📦 Cached Load Times (Top 3):")
sorted_cached = sort(collect(cached_times), by=x->x[2], rev=true)
for (i, (pkg, time)) in enumerate(sorted_cached[1:min(3, length(sorted_cached))])
    println("   $i. $(rpad(pkg, 20)) $(round(time, digits=3))s")
end

total_cached = sum(values(cached_times))
println("\n📈 Summary:")
println("   Main NonlinearSolve (cached): $(round(main_cached_time, digits=3))s")
println("   Total deps (cached):          $(round(total_cached, digits=3))s")

# Check compilation artifacts
depot_path = first(DEPOT_PATH)
compiled_path = joinpath(depot_path, "compiled", "v$(VERSION.major).$(VERSION.minor)")

if isdir(compiled_path)
    # Count .ji files
    ji_files = []
    for (root, dirs, files) in walkdir(compiled_path)
        for file in files
            if endswith(file, ".ji")
                push!(ji_files, file)
            end
        end
    end
    
    # Get sizes
    total_size = 0
    for file in ji_files
        try
            path = joinpath(compiled_path, file)
            if isfile(path)
                total_size += stat(path).size
            end
        catch
        end
    end
    
    println("\n💾 Compilation Artifacts:")
    println("   Compiled cache files:        $(length(ji_files))")
    println("   Total cache size:            $(round(total_size / 1024^2, digits=2)) MB")
    
    # Find NonlinearSolve related files
    nl_files = filter(f -> contains(f, "NonlinearSolve"), ji_files)
    if !isempty(nl_files)
        println("   NonlinearSolve cache files:  $(length(nl_files))")
        for file in nl_files[1:min(5, length(nl_files))]
            println("      - $file")
        end
        if length(nl_files) > 5
            println("      ... ($(length(nl_files) - 5) more)")
        end
    end
end

println("\n✅ Precompile analysis complete!")