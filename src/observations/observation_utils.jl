function compute_sectioned_observation(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, obs_strategy::AbstractObservationStrategy) where {T, A, O, RW, V, OBS, M, RS, C}
    prob = env.prob
    N = prob.params.N
    current_u = @view env.state[1:N]
    current_λ = @view env.state[(N + 1):end]
    max_shocks = T(6)
    max_pressure = T(6)

    dx = RDE.get_dx(prob)::T
    minisections = obs_strategy.minisections
    minisection_size = N ÷ minisections

    minisection_observations_u = get_minisection_observations(current_u, minisection_size) ./ max_pressure
    minisection_observations_λ = get_minisection_observations(current_λ, minisection_size)

    shocks::T = RDE.count_shocks(current_u, dx) / max_shocks
    target_shock_count::T = obs_strategy.target_shock_count / max_shocks

    return minisection_observations_u, minisection_observations_λ, shocks, target_shock_count
end

function get_observable_minisections(obs_strategy)
    n_sections = obs_strategy.n_sections
    L = obs_strategy.L
    v = obs_strategy.look_ahead_speed
    dt = obs_strategy.dt
    m = obs_strategy.minisections_per_section

    observable_minisections = Int(floor(m * (1 + n_sections * v * dt / L)))
    return observable_minisections
end

function get_minisection_observations(data, minisection_size)
    minisection_u = reshape(data, minisection_size, :)
    minisection_observations = RDE.turbo_column_maximum(minisection_u)
    return minisection_observations
end
