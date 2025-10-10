struct MeanInjectionPressureObservation <: AbstractObservationStrategy end

function compute_observation(env::RDEEnv{T, A, O, R, V, OBS}, strategy::MeanInjectionPressureObservation) where {T, A, O, R, V, OBS}
    return [mean(env.prob.method.cache.u_p_current)]
end

function get_init_observation(strategy::MeanInjectionPressureObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    return Vector{T}(undef, 1)
end
