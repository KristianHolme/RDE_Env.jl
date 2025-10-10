function compute_observation(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, strategy::SectionedStateObservation) where {T, A, O, RW, V, OBS, M, RS, C}
    (
        minisection_observations_u::Vector{T},
        minisection_observations_λ::Vector{T},
        shocks_normalized::T,
        target_shock_count_normalized::T,
    ) = compute_sectioned_observation(env, strategy)
    return vcat(minisection_observations_u, minisection_observations_λ, shocks_normalized, target_shock_count_normalized)
end

function get_init_observation(strategy::SectionedStateObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    return Vector{T}(undef, 2 * strategy.minisections + 2)
end
