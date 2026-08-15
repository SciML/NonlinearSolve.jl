"""
    LocalEnsemble(npoints; nchunks = Threads.nthreads())

A partition of the `npoints` local problems into `nchunks` contiguous chunks, for use with
[`ensemble_foreach`](@ref). The partition is fixed at construction, so a threaded run visits
the same points in the same groups every time — which is what makes a run reproducible.

Chunks, not threads, are the unit of work: a task may migrate between threads mid-run, so
`Threads.threadid()` is never a valid index into per-worker storage. Size per-chunk
workspaces by `length(ensemble.chunks)` instead.
"""
struct LocalEnsemble
    npoints::Int
    chunks::Vector{UnitRange{Int}}
end

function LocalEnsemble(npoints::Int; nchunks::Int = Threads.nthreads())
    npoints ≥ 0 || throw(ArgumentError("`npoints` must be non-negative; got $(npoints)."))
    nchunks = clamp(nchunks, 1, max(npoints, 1))
    base, rem = divrem(npoints, nchunks)
    chunks = Vector{UnitRange{Int}}(undef, nchunks)
    stop = 0
    for i in 1:nchunks
        start = stop + 1
        stop += base + (i ≤ rem)
        chunks[i] = start:stop
    end
    return LocalEnsemble(npoints, chunks)
end

nchunks(ensemble::LocalEnsemble) = length(ensemble.chunks)

"""
    ensemble_foreach(f, ensemble, args...; threaded = true)

Run `f(chunk::UnitRange{Int}, ichunk::Int, args...)` once per chunk of `ensemble`, in
parallel when `threaded`.

`args` are forwarded by value, which is how the current local tolerance (read once on the
host with [`local_tolerance`](@ref)) reaches the local solves.

`f` must only write to storage owned by its own chunk: the per-point internal variables of
the points in `chunk`, and the `ichunk`-th slice of any scratch. Anything with overlapping
writes — the scatter-add of `R̄` or `S` over shared degrees of freedom — belongs *outside*
this call, run serially, or the result depends on how the additions happened to be grouped.
Under that rule repeated threaded runs agree bitwise, and agree with a serial run up to
rounding of the reduction that follows.
"""
function ensemble_foreach(
        f::F, ensemble::LocalEnsemble, args::Vararg{Any, N}; threaded::Bool = true
    ) where {F, N}
    chunks = ensemble.chunks
    if !threaded || length(chunks) == 1 || Threads.nthreads() == 1
        for (ichunk, chunk) in enumerate(chunks)
            f(chunk, ichunk, args...)
        end
        return nothing
    end
    tasks = Vector{Task}(undef, length(chunks))
    for (ichunk, chunk) in enumerate(chunks)
        tasks[ichunk] = Threads.@spawn f(chunk, ichunk, args...)
    end
    foreach(wait, tasks)
    return nothing
end

"""
    LocalStateBuffer(committed)

Double buffer for the internal variables of a local ensemble: `committed` holds the state of
the last accepted global iterate, `scratch` the state a trial is building.

The split is what lets every residual evaluation be a trial. Trials only ever read
`committed` — so a line search's `ϕ(α)` depends on `α` alone and not on which trials ran
before it — and write to `scratch`. It is also a correctness requirement, not just an
efficiency one, whenever a local problem has several roots (return-mapping plasticity,
damage): the warm start is what selects the branch, so a trial that overwrote the committed
state would change which root later trials converge to.

The point index is the last dimension, e.g. `committed` is `n_internal × npoints`.
"""
struct LocalStateBuffer{A <: AbstractArray}
    committed::A
    scratch::A
end

LocalStateBuffer(committed::AbstractArray) = LocalStateBuffer(committed, copy(committed))

"The committed internal state of point `i`; read-only during a trial."
@inline committed_state(buffer::LocalStateBuffer, i::Int) =
    selectdim(buffer.committed, ndims(buffer.committed), i)

"""
    trial_state(buffer, i)

The scratch internal state of point `i`, warm-started from its committed state. Solve the
local problem into the returned view.
"""
@inline function trial_state(buffer::LocalStateBuffer, i::Int)
    scratch = selectdim(buffer.scratch, ndims(buffer.scratch), i)
    copyto!(scratch, committed_state(buffer, i))
    return scratch
end

"""
    commit_local_state!(buffer)

Promote the scratch state to committed. Call this from `commit_internal!`, i.e. exactly once
per accepted global iterate, after re-running the local solves at that iterate.
"""
function commit_local_state!(buffer::LocalStateBuffer)
    copyto!(buffer.committed, buffer.scratch)
    return buffer
end
