using Test

# Preferences for TimerOutputs are read at precompile time, so enabling them requires a
# fresh process that loads NonlinearSolveBase with the preference already set (#1224).
const NSB_UUID = "be0214bd-f91f-a760-ac4e-3421ce2b2da0"
const NSB_PATH = dirname(@__DIR__)

mktempdir() do dir
    open(joinpath(dir, "Project.toml"), "w") do io
        println(io, """
        [deps]
        NonlinearSolveBase = "$(NSB_UUID)"
        Preferences = "21216c6a-2e73-6563-6e65-726566657250"
        TimerOutputs = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
        UUIDs = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
        """)
    end
    script = joinpath(dir, "check.jl")
    write(script, """
        using Pkg
        Pkg.develop(PackageSpec(; path = raw"$(NSB_PATH)"))
        using Preferences, UUIDs
        set_preferences!(UUID("$(NSB_UUID)"), "enable_timer_outputs" => true; force = true)
        # Preference changes invalidate the cache; force a fresh compile against it.
        Base.compilecache(Base.PkgId(UUID("$(NSB_UUID)"), "NonlinearSolveBase"))
        using NonlinearSolveBase, TimerOutputs
        to = NonlinearSolveBase.get_timer_output()
        to isa TimerOutput || error("expected TimerOutput, got \$(typeof(to))")
        @eval NonlinearSolveBase begin
            function __timer_probe(to)
                @static_timeit to "probe" begin
                    return 1 + 1
                end
            end
        end
        NonlinearSolveBase.__timer_probe(to) == 2 || error("probe returned wrong value")
        TimerOutputs.ncalls(to["probe"]) == 1 || error("probe was not timed")
        println("timer_outputs_enabled: ok")
        """)
    @test success(run(pipeline(`$(Base.julia_cmd()) --project=$(dir) $(script)`;
        stdout = stdout, stderr = stderr)))
end
