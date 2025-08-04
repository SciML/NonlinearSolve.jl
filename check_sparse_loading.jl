#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

println("Checking if SparseArrays gets loaded automatically...")

# Check loaded modules before loading NonlinearSolve
modules_before = names(Main, imported=true)
loaded_pkgs_before = collect(keys(Base.loaded_modules))

println("📋 Modules before loading NonlinearSolve: $(length(modules_before))")

# Load NonlinearSolve
println("📦 Loading NonlinearSolve...")
load_time = @elapsed using NonlinearSolve

# Check what got loaded
modules_after = names(Main, imported=true)
loaded_pkgs_after = collect(keys(Base.loaded_modules))

new_modules = setdiff(modules_after, modules_before)
new_packages = setdiff(loaded_pkgs_after, loaded_pkgs_before)

println("   ⏱️  Load time: $(round(load_time, digits=3))s")
println("   📋 New modules: $(length(new_modules))")
println("   📦 New packages: $(length(new_packages))")

# Check specifically for SparseArrays
sparse_loaded = any(pkg -> contains(string(pkg.name), "SparseArrays"), new_packages)
println("   🔍 SparseArrays loaded automatically: $sparse_loaded")

if sparse_loaded
    println("   ⚠️  SparseArrays was loaded - extension system may be triggering it")
    # Find which package caused SparseArrays to load
    for pkg in new_packages
        if contains(string(pkg.name), "SparseArrays")
            println("      📌 Found: $pkg")
        end
    end
else
    println("   ✅ SparseArrays was NOT loaded automatically")
end

# Check which extensions are loaded
println("\n🔌 Loaded extensions:")
extension_count = 0
for (name, mod) in Base.loaded_modules
    if contains(string(name), "Ext")
        extension_count += 1
        if extension_count <= 10  # Show first 10
            println("   $extension_count. $name")
        end
    end
end
if extension_count > 10
    println("   ... ($(extension_count - 10) more extensions)")
end

println("\n✅ Analysis complete!")