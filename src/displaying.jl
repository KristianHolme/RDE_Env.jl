#TODO: move to appropriate files
function Base.show(io::IO, obs::SampledStateObservation)
    return print(io, "SampledStateObservation(n_samples=$(obs.n_samples))")
end

function Base.show(io::IO, ::MIME"text/plain", obs::SampledStateObservation)
    println(io, "SampledStateObservation:")
    println(io, "  n_samples: $(obs.n_samples)")
    return nothing
end

function Base.show(io::IO, obs::SectionedStateObservation)
    print(io, "SectionedStateObservation(minisections=$(obs.minisections))")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", obs::SectionedStateObservation)
    println(io, "SectionedStateObservation:")
    println(io, "  minisections: $(obs.minisections)")
    return nothing
end

function Base.show(io::IO, ::StateObservation)
    print(io, "StateObservation()")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", ::StateObservation)
    println(io, "StateObservation: returns full state")
    return nothing
end

function Base.show(io::IO, obs::FourierObservation)
    print(io, "FourierObservation(fft_terms=$(obs.fft_terms))")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", obs::FourierObservation)
    println(io, "FourierObservation:")
    println(io, "  fft_terms: $(obs.fft_terms)")
    return nothing
end


function Base.show(io::IO, a::ScalarAreaScalarPressureAction)
    print(io, "ScalarAreaScalarPressureAction()")
    return nothing
end

function Base.show(io::IO, a::VectorPressureAction)
    print(io, "VectorPressureAction(n_sections=$(a.n_sections))")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", a::VectorPressureAction)
    println(io, "VectorPressureAction:")
    println(io, "  n_sections: $(a.n_sections)")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", rt::ShockSpanReward)
    println(io, "ShockSpanReward:")
    println(io, "  span_scale: $(rt.span_scale)")
    println(io, "  shock_weight: $(rt.shock_weight)")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", rt::ShockPreservingReward)
    println(io, "ShockPreservingReward:")
    println(io, "  span_scale: $(rt.span_scale)")
    println(io, "  shock_weight: $(rt.shock_weight)")
    return println(io, "  abscence_limit: $(rt.abscence_limit)")
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
    println(io, "  goal strategy:")
    return println(io, "  steps taken: $(env.steps_taken)")
end
