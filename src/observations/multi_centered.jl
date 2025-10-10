@kwdef struct MultiCenteredObservation <: AbstractMultiAgentObservationStrategy
    n_sections::Int = 4
    target_shock_count::Int = 4
    minisections::Int = 32
end
MultiCenteredObservation(n::Int) = MultiCenteredObservation(n_sections = n)

function Base.show(io::IO, obs_strategy::MultiCenteredObservation)
    return if get(io, :compact, false)::Bool
        print(io, "MultiCenteredObservation(n_sections=$(obs_strategy.n_sections))")
    else
        print(io, "MultiCenteredObservation(n_sections=$(obs_strategy.n_sections), target_shock_count=$(obs_strategy.target_shock_count))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", obs_strategy::MultiCenteredObservation)
    println(io, "MultiCenteredObservation:")
    println(io, "  n_sections: $(obs_strategy.n_sections)")
    println(io, "  target_shock_count: $(obs_strategy.target_shock_count)")
    return println(io, "  minisections: $(obs_strategy.minisections)")
end

function compute_observation(env::RDEEnv{T, A, O, R, V, OBS}, obs_strategy::MultiCenteredObservation) where {T, A, O, R, V, OBS}
    (
        minisection_observations_u,
        minisection_observations_λ,
        shocks,
        target_shock_count,
    ) = compute_sectioned_observation(env, obs_strategy)

    obs_length = obs_strategy.minisections * 2 + 2
    n_sections = obs_strategy.n_sections
    minisections = obs_strategy.minisections
    minisections_per_section = minisections ÷ n_sections
    result = Matrix{T}(undef, obs_length, n_sections)

    for i in 1:n_sections
        result[1:minisections, i] .= circshift(minisection_observations_u, -minisections_per_section * (i - 1))
        result[(minisections + 1):(minisections * 2), i] .= circshift(minisection_observations_λ, -minisections_per_section * (i - 1))
        result[end - 1, i] = T(shocks)
        result[end, i] = T(target_shock_count)
    end

    return result
end

function get_init_observation(obs_strategy::MultiCenteredObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    obs_dim = obs_strategy.minisections * 2 + 2
    return Matrix{T}(undef, obs_dim, obs_strategy.n_sections)
end
