struct FullStateObservation <: AbstractObservationStrategy end

initialize_cache(::FullStateObservation, N::Int, ::Type{T}) where {T} = NoCache()

function compute_observation!(obs, env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, ::FullStateObservation) where {T, A, O, RW, G, V, OBS, M, RS, C}
    N = length(env.state) ÷ 2
    obs_u = @view obs[1:N]
    obs_λ = @view obs[(N + 1):(2 * N)]
    u = @view env.state[1:N]
    λ = @view env.state[(N + 1):end]
    obs_u .= u
    obs_λ .= λ
    return obs
end

function get_init_observation(::FullStateObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    return Vector{T}(undef, 2N)
end

@kwdef struct FullStateCenteredObservation <: AbstractMultiAgentObservationStrategy
    n_sections::Int = 4
end

initialize_cache(::FullStateCenteredObservation, N::Int, ::Type{T}) where {T} = NoCache()

function Base.show(io::IO, obs_strategy::FullStateCenteredObservation)
    return print(io, "FullStateCenteredObservation(n_sections=$(obs_strategy.n_sections))")
end

function Base.show(io::IO, ::MIME"text/plain", obs_strategy::FullStateCenteredObservation)
    println(io, "FullStateCenteredObservation:")
    return println(io, "  n_sections: $(obs_strategy.n_sections)")
end

function compute_observation!(obs, env::RDEEnv{T, A, O, R, G, V, OBS}, obs_strategy::FullStateCenteredObservation) where {T, A, O, R, G, V, OBS}
    n_sections = obs_strategy.n_sections
    N = env.prob.params.N
    points_per_section = N ÷ n_sections
    center_offset = points_per_section ÷ 2
    target_center = (N ÷ 2) + 1

    u = @view env.state[1:N]
    λ = @view env.state[(N + 1):end]

    for i in 1:n_sections
        section_start = (i - 1) * points_per_section + 1
        section_center = section_start + center_offset
        shift = target_center - section_center

        obs_u_view = @view obs[1:N, i]
        obs_λ_view = @view obs[(N + 1):(2 * N), i]
        circshift!(obs_u_view, u, shift)
        circshift!(obs_λ_view, λ, shift)
    end

    return obs
end

function get_init_observation(obs_strategy::FullStateCenteredObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    return Matrix{T}(undef, 2 * N, obs_strategy.n_sections)
end
