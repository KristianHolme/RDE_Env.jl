function Base.show(io::IO, ::FullStateObservation)
    print(io, "FullStateObservation()")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", ::FullStateObservation)
    println(io, "FullStateObservation: returns raw [u; λ]")
    return nothing
end

function Base.show(io::IO, env::RDEEnv{T}) where {T <: AbstractFloat}
    return if get(io, :compact, false)::Bool
        print(io, "RDEEnv{$T}(t=$(env.t), steps=$(env.steps_taken))")
    else
        print(io, "RDEEnv{$T}(t=$(env.t), steps=$(env.steps_taken), $(env.action_strat))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", env::RDEEnv{T}) where {T <: AbstractFloat}
    println(io, "RDEEnv{$T}:")
    println(io, "  dt: $(env.dt)")
    println(io, "  t: $(env.t)")
    println(io, "  truncated: $(env.truncated)")
    println(io, "  terminated: $(env.terminated)")
    println(io, "  action strategy: $(env.action_strat)")
    println(io, "  observation strategy: $(env.observation_strat)")
    println(io, "  reward type: $(env.reward_strat)")
    println(io, "  context strategy: $(env.context_strat)")
    return println(io, "  steps taken: $(env.steps_taken)")
end
