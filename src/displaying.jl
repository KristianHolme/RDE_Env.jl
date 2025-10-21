function Base.show(io::IO, obs::SampledStateObservation)
    return print(io, "SampledStateObservation(n_samples=$(obs.n_samples))")
end

function Base.show(io::IO, ::MIME"text/plain", obs::SampledStateObservation)
    println(io, "SampledStateObservation:")
    return println(io, "  n_samples: $(obs.n_samples)")
end

function Base.show(io::IO, obs::SectionedStateObservation)
    return print(io, "SectionedStateObservation(minisections=$(obs.minisections))")
end

function Base.show(io::IO, ::MIME"text/plain", obs::SectionedStateObservation)
    println(io, "SectionedStateObservation:")
    println(io, "  minisections: $(obs.minisections)")
    return println(io, "  target_shock_count: $(obs.target_shock_count)")
end

function Base.show(io::IO, ::StateObservation)
    return print(io, "StateObservation()")
end

function Base.show(io::IO, ::MIME"text/plain", ::StateObservation)
    return println(io, "StateObservation: returns full state")
end

function Base.show(io::IO, obs::FourierObservation)
    return print(io, "FourierObservation(fft_terms=$(obs.fft_terms))")
end

function Base.show(io::IO, ::MIME"text/plain", obs::FourierObservation)
    println(io, "FourierObservation:")
    return println(io, "  fft_terms: $(obs.fft_terms)")
end

function Base.show(io::IO, a::ScalarPressureAction)
    return print(io, "ScalarPressureAction()")
end

function Base.show(io::IO, ::MIME"text/plain", a::ScalarPressureAction)
    println(io, "ScalarPressureAction:")
    return println(io, "  N: $(a.N)")
end

function Base.show(io::IO, a::ScalarAreaScalarPressureAction)
    return print(io, "ScalarAreaScalarPressureAction()")
end

function Base.show(io::IO, ::MIME"text/plain", a::ScalarAreaScalarPressureAction)
    println(io, "ScalarAreaScalarPressureAction:")
    return println(io, "  N: $(a.N)")
end

function Base.show(io::IO, a::VectorPressureAction)
    return print(io, "VectorPressureAction(n_sections=$(a.n_sections))")
end

function Base.show(io::IO, ::MIME"text/plain", a::VectorPressureAction)
    println(io, "VectorPressureAction:")
    println(io, "  n_sections: $(a.n_sections)")
    return println(io, "  N: $(a.N)")
end

function Base.show(io::IO, rt::ShockSpanReward)
    return print(io, "ShockSpanReward(target_shock_count=$(rt.target_shock_count))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::ShockSpanReward)
    println(io, "ShockSpanReward:")
    println(io, "  target_shock_count: $(rt.target_shock_count)")
    println(io, "  span_scale: $(rt.span_scale)")
    return println(io, "  shock_weight: $(rt.shock_weight)")
end

function Base.show(io::IO, rt::ShockPreservingReward)
    return print(io, "ShockPreservingReward(target_shock_count=$(rt.target_shock_count))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::ShockPreservingReward)
    println(io, "ShockPreservingReward:")
    println(io, "  target_shock_count: $(rt.target_shock_count)")
    println(io, "  span_scale: $(rt.span_scale)")
    println(io, "  shock_weight: $(rt.shock_weight)")
    return println(io, "  abscence_limit: $(rt.abscence_limit)")
end

function Base.show(io::IO, rt::ShockPreservingSymmetryReward)
    return print(io, "ShockPreservingSymmetryReward(target_shock_count=$(rt.target_shock_count))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::ShockPreservingSymmetryReward)
    println(io, "ShockPreservingSymmetryReward:")
    println(io, "  target_shock_count: $(rt.target_shock_count)")
    return println(io, "  cache size: $(length(rt.cache))")
end

function Base.show(io::IO, cache::RDEEnvCache)
    return if get(io, :compact, false)::Bool
        print(io, "RDEEnvCache{$(eltype(cache.circ_u))}")
    else
        print(io, "RDEEnvCache{$(eltype(cache.circ_u))}(N=$(length(cache.circ_u)))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", cache::RDEEnvCache)
    println(io, "RDEEnvCache{$(eltype(cache.circ_u))} with:")
    println(io, "  circ_u: $(typeof(cache.circ_u)) of size $(length(cache.circ_u))")
    println(io, "  circ_λ: $(typeof(cache.circ_λ)) of size $(length(cache.circ_λ))")
    println(io, "  prev_u: $(typeof(cache.prev_u)) of size $(length(cache.prev_u))")
    println(io, "  prev_λ: $(typeof(cache.prev_λ)) of size $(length(cache.prev_λ))")
    return println(io, "  action: $(typeof(cache.action)) of size $(size(cache.action))")
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
    println(io, "  action type: $(env.action_strat)")
    println(io, "  observation strategy: $(env.observation_strategy)")
    println(io, "  reward type: $(env.reward_type)")
    return println(io, "  steps taken: $(env.steps_taken)")
end
