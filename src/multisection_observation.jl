@kwdef struct MultiSectionObservation <: AbstractObservationStrategy
    n_sections::Int = 4
    look_ahead_speed::Float32 = 1.65f0
    minisections_per_section::Int = 8
    dt::Float32 = 0.5f0
    L::Float32 = 2f0*π
end

function Base.show(io::IO, obs_strategy::MultiSectionObservation)
    print(io, "MultiSectionObservation(n_sections=$(obs_strategy.n_sections), look_ahead_speed=$(obs_strategy.look_ahead_speed), minisections_per_section=$(obs_strategy.minisections_per_section), dt=$(obs_strategy.dt), L=$(obs_strategy.L))")
end

function compute_observation(env::AbstractRDEEnv, obs_strategy::MultiSectionObservation)
    N = env.prob.params.N
    current_u = @view env.state[1:N]
    env.cache.circ_u[:] .= current_u
    circ_u = env.cache.circ_u
    max_shocks = 6f0
    max_pressure = 6f0

    n_sections = obs_strategy.n_sections
    dx = env.prob.x[2] - env.prob.x[1]
    m = obs_strategy.minisections_per_section
    section_size = N ÷ n_sections
    minisection_size = section_size ÷ m

    
    observable_minisections = get_observable_minisections(obs_strategy)

    minisection_observations = get_minisection_observations(circ_u, minisection_size)

    last_minisection_in_sections = collect(m:m:m*n_sections)


    #concsiously not normalizing, leave it to PPO
    # observations = map(last_minisection_in_sections) do i
    #     @view minisection_observations[(i-observable_minisections+1):i]
    # end

    shocks = RDE.count_shocks(current_u, dx)
    target_shock_count = env.reward_type.target_shock_count
    span = maximum(current_u) - minimum(current_u)

    # Pre-allocate the final matrix
    obs_length = observable_minisections + 3  # +3 for shocks, target_shock_count, span
    result = Matrix{Float32}(undef, obs_length, n_sections)
    
    # Fill the matrix directly
    for i in 1:n_sections
        # Copy the observation part
        last_minisection = last_minisection_in_sections[i]
        result[1:end-3, i] .= minisection_observations[(last_minisection-observable_minisections+1):last_minisection] ./ max_pressure
        # Add the additional values
        result[end-2, i] = shocks / max_shocks
        result[end-1, i] = target_shock_count / max_shocks
        result[end, i] = span / max_pressure
    end
    
    return result
end

function get_observable_minisections(obs_strategy::MultiSectionObservation)
    n_sections = obs_strategy.n_sections
    L = obs_strategy.L
    v = obs_strategy.look_ahead_speed
    dt = obs_strategy.dt
    m = obs_strategy.minisections_per_section

    
    observable_minisections = Int(floor(m*(1+n_sections*v*dt/L)))
    return observable_minisections
end

function get_minisection_observations(circ_u, minisection_size)
    minisection_u = reshape(circ_u, minisection_size, :)
    minisection_observations = maximum(minisection_u, dims=1)
    return minisection_observations
end

function get_init_observation(obs_strategy::MultiSectionObservation, N::Int)
    obs_dim = get_observable_minisections(obs_strategy) + 3
    return Matrix{Float32}(undef, obs_dim, obs_strategy.n_sections)
end
