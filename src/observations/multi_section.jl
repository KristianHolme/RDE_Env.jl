@kwdef struct MultiSectionObservation <: AbstractMultiAgentObservationStrategy
    n_sections::Int = 4
    target_shock_count::Int = 3
    look_ahead_speed::Float32 = 1.65f0
    minisections_per_section::Int = 8
    dt::Float32 = 1.0f0
    L::Float32 = 2.0f0 * π
end
MultiSectionObservation(n::Int) = MultiSectionObservation(n_sections = n)

function Base.show(io::IO, obs_strategy::MultiSectionObservation)
    return if get(io, :compact, false)::Bool
        print(io, "MultiSectionObservation(n_sections=$(obs_strategy.n_sections))")
    else
        print(io, "MultiSectionObservation(n_sections=$(obs_strategy.n_sections), minisections_per_section=$(obs_strategy.minisections_per_section))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", obs_strategy::MultiSectionObservation)
    println(io, "MultiSectionObservation:")
    println(io, "  n_sections: $(obs_strategy.n_sections)")
    println(io, "  look_ahead_speed: $(obs_strategy.look_ahead_speed)")
    println(io, "  minisections_per_section: $(obs_strategy.minisections_per_section)")
    println(io, "  dt: $(obs_strategy.dt)")
    return println(io, "  L: $(obs_strategy.L)")
end

function compute_observation(env::RDEEnv{T, A, O, R, V, OBS}, obs_strategy::MultiSectionObservation) where {T, A, O, R, V, OBS}
    N = env.prob.params.N
    current_u = @view env.state[1:N]
    current_λ = @view env.state[(N + 1):end]
    max_shocks = T(6)
    max_pressure = T(6)

    n_sections = obs_strategy.n_sections
    dx = env.prob.x[2] - env.prob.x[1]
    m = obs_strategy.minisections_per_section
    section_size = N ÷ n_sections
    minisection_size = section_size ÷ m

    observable_minisections = get_observable_minisections(obs_strategy)

    minisection_observations_u = get_minisection_observations(current_u, minisection_size)
    minisection_observations_λ = get_minisection_observations(current_λ, minisection_size)
    last_minisection_in_sections = collect(m:m:(m * n_sections))

    shocks = RDE.count_shocks(current_u, dx)
    target_shock_count = obs_strategy.target_shock_count
    u_min, u_max = RDE.turbo_extrema(current_u)
    span = u_max - u_min

    obs_length = observable_minisections * 2 + 3
    result = Matrix{T}(undef, obs_length, n_sections)

    for i in 1:n_sections
        last_minisection = last_minisection_in_sections[i]
        result[1:observable_minisections, i] .= minisection_observations_u[(last_minisection - observable_minisections + 1):last_minisection] ./ max_pressure
        result[(observable_minisections + 1):(observable_minisections * 2), i] .= minisection_observations_λ[(last_minisection - observable_minisections + 1):last_minisection]
        result[end - 2, i] = T(shocks / max_shocks)
        result[end - 1, i] = T(target_shock_count / max_shocks)
        result[end, i] = T(span / max_pressure)
    end

    return result
end

function get_init_observation(obs_strategy::MultiSectionObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    obs_dim = get_observable_minisections(obs_strategy) * 2 + 3
    return Matrix{T}(undef, obs_dim, obs_strategy.n_sections)
end
