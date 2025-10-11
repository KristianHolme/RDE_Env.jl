struct FourierObservation <: AbstractObservationStrategy
    fft_terms::Int
end


struct StateObservation <: AbstractObservationStrategy end


@kwdef struct SectionedStateObservation <: AbstractObservationStrategy
    minisections::Int = 32
    target_shock_count::Int = 3
end


struct SampledStateObservation <: AbstractObservationStrategy
    n_samples::Int
end
function compute_observation!(obs, env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, strategy::FourierObservation) where {T, A, O, RW, V, OBS, M, RS, C}
    N = env.prob.params.N

    current_u = @view env.state[1:N]
    current_λ = @view env.state[(N + 1):end]

    # env.cache.circ_u[:] .= current_u
    # env.cache.circ_λ[:] .= current_λ

    fft_u = abs.(fft(current_u)) #was circ_u
    fft_λ = abs.(fft(current_λ)) #was circ_λ

    n_terms = min(strategy.fft_terms, N ÷ 2 + 1)

    # Create views directly into obs for each component
    u_obs = @view obs[1:n_terms]
    λ_obs = @view obs[(n_terms + 1):(2 * n_terms)]

    # Get views of relevant FFT terms (avoiding intermediate allocation)
    u_terms = @view fft_u[1:n_terms]
    λ_terms = @view fft_λ[1:n_terms]

    # Min-max normalization to [0,1] range, writing directly into obs
    u_terms_min = RDE.turbo_minimum(u_terms)
    u_terms_max = RDE.turbo_maximum(u_terms)
    λ_terms_min = RDE.turbo_minimum(λ_terms)
    λ_terms_max = RDE.turbo_maximum(λ_terms)

    u_range_inv = inv(u_terms_max - u_terms_min + T(1.0e-8))
    λ_range_inv = inv(λ_terms_max - λ_terms_min + T(1.0e-8))

    # Normalize and scale to [-1,1] in one pass
    @. u_obs = 2 * (u_terms - u_terms_min) * u_range_inv - 1
    @. λ_obs = 2 * (λ_terms - λ_terms_min) * λ_range_inv - 1

    # Control parameters already naturally bounded - write directly into obs
    obs[2 * n_terms + 1] = mean(env.prob.method.cache.s_current) / env.smax
    obs[2 * n_terms + 2] = mean(env.prob.method.cache.u_p_current) / env.u_pmax

    return obs
end


function compute_observation!(obs, env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, rt::StateObservation) where {T, A, O, RW, V, OBS, M, RS, C}
    N = length(env.state) ÷ 2
    u = @view env.state[1:N]
    λ = @view env.state[(N + 1):end]

    ϵ = T(1.0e-8)
    u_max_inv = inv(max(RDE.turbo_maximum_abs(u), ϵ))
    λ_max_inv = inv(max(RDE.turbo_maximum_abs(λ), ϵ))

    # Create views into obs for each component
    obs_u = @view obs[1:N]
    obs_λ = @view obs[(N + 1):(2 * N)]

    # Normalize directly into obs
    @. obs_u = u * u_max_inv
    @. obs_λ = λ * λ_max_inv

    # Write control parameters directly to the last two positions
    obs[2 * N + 1] = mean(env.prob.method.cache.s_current) / env.smax
    obs[2 * N + 2] = mean(env.prob.method.cache.u_p_current) / env.u_pmax

    return obs
end

function compute_sectioned_observation!(minisection_observations_u, minisection_observations_λ, env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, obs_strategy::AbstractObservationStrategy) where {T, A, O, RW, V, OBS, M, RS, C}
    prob = env.prob #::RDEProblem{T, M, RS, C}
    N = prob.params.N
    current_u = @view env.state[1:N]
    current_λ = @view env.state[(N + 1):end]
    # env.cache.circ_u[:] .= current_u
    # env.cache.circ_λ[:] .= current_λ
    # circ_u = env.cache.circ_u
    # circ_λ = env.cache.circ_λ
    max_shocks = T(6)
    #TODO: change to 3?
    max_pressure = T(6)

    dx = RDE.get_dx(prob)::T
    minisections = obs_strategy.minisections
    minisection_size = N ÷ minisections

    get_minisection_observations!(minisection_observations_u, current_u, minisection_size)
    @. minisection_observations_u /= max_pressure
    get_minisection_observations!(minisection_observations_λ, current_λ, minisection_size)

    shocks::T = RDE.count_shocks(current_u, dx) / max_shocks
    target_shock_count::T = obs_strategy.target_shock_count / max_shocks

    return shocks, target_shock_count
end
function compute_observation!(obs, env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, strategy::SectionedStateObservation) where {T, A, O, RW, V, OBS, M, RS, C}
    minisections = strategy.minisections

    # Create views into obs for each component
    minisection_observations_u = @view obs[1:minisections]
    minisection_observations_λ = @view obs[(minisections + 1):(2 * minisections)]

    shocks_normalized, target_shock_count_normalized = compute_sectioned_observation!(
        minisection_observations_u, minisection_observations_λ, env, strategy
    )

    # Write scalar values directly to the end of obs
    obs[2 * minisections + 1] = shocks_normalized
    obs[2 * minisections + 2] = target_shock_count_normalized

    return obs
end

function compute_observation!(obs, env::RDEEnv{T, A, O, R, V, OBS}, strategy::SampledStateObservation) where {T, A, O, R, V, OBS}
    N = env.prob.params.N
    n = strategy.n_samples

    u = @view env.state[1:N]
    λ = @view env.state[(N + 1):end]

    # Create views into obs for each component
    sampled_u = @view obs[1:n]
    sampled_λ = @view obs[(n + 1):(2 * n)]

    # Sample the data directly into obs (avoiding indices array allocation)
    step = (N - 1) / (n - 1)
    for i in 1:n
        idx = round(Int, 1 + (i - 1) * step)
        sampled_u[i] = u[idx]
        sampled_λ[i] = λ[idx]
    end

    ϵ = T(1.0e-8)
    u_max_inv = inv(max(RDE.turbo_maximum_abs(sampled_u), ϵ))
    λ_max_inv = inv(max(RDE.turbo_maximum_abs(sampled_λ), ϵ))

    # Normalize in place
    @. sampled_u *= u_max_inv
    @. sampled_λ *= λ_max_inv

    # Write normalized time directly to obs
    obs[2 * n + 1] = env.t / env.prob.params.tmax

    return obs
end

#TODO fix this, use another approach to get number of elements and then init vector
function get_init_observation(strategy::FourierObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    n_terms = min(strategy.fft_terms, N ÷ 2 + 1)
    return Vector{T}(undef, n_terms * 2 + 2)
end

function get_init_observation(::StateObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    return Vector{T}(undef, 2N + 2)
end

function get_init_observation(strategy::SampledStateObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    return Vector{T}(undef, 2 * strategy.n_samples + 1)
end

function get_init_observation(strategy::SectionedStateObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    return Vector{T}(undef, 2 * strategy.minisections + 2)
end

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

function compute_observation!(obs, env::RDEEnv{T, A, O, R, V, OBS}, obs_strategy::MultiSectionObservation) where {T, A, O, R, V, OBS}
    N = env.prob.params.N
    current_u = @view env.state[1:N]
    current_λ = @view env.state[(N + 1):end]
    # env.cache.circ_u[:] .= current_u
    # circ_u = env.cache.circ_u
    # circ_λ = env.cache.circ_λ
    max_shocks = T(6)
    max_pressure = T(6)
    max_pressure_inv = inv(max_pressure)
    max_shocks_inv = inv(max_shocks)

    n_sections = obs_strategy.n_sections
    dx = env.prob.x[2] - env.prob.x[1]
    m = obs_strategy.minisections_per_section
    section_size = N ÷ n_sections
    minisection_size = section_size ÷ m

    observable_minisections = get_observable_minisections(obs_strategy)

    # Allocate temporary buffers for all minisection observations
    # (these need to be full-length to compute all minisections once)
    n_total_minisections = N ÷ minisection_size
    minisection_observations_u = Vector{T}(undef, n_total_minisections)
    minisection_observations_λ = Vector{T}(undef, n_total_minisections)

    get_minisection_observations!(minisection_observations_u, current_u, minisection_size)
    get_minisection_observations!(minisection_observations_λ, current_λ, minisection_size)

    #concsiously not normalizing, leave it to PPO
    # observations = map(last_minisection_in_sections) do i
    #     @view minisection_observations[(i-observable_minisections+1):i]
    # end

    shocks = RDE.count_shocks(current_u, dx)
    target_shock_count = obs_strategy.target_shock_count
    u_min, u_max = RDE.turbo_extrema(current_u)
    span = u_max - u_min

    # Fill the matrix directly
    for i in 1:n_sections
        # Calculate last minisection index directly (avoiding collect)
        last_minisection = m * i

        # Get views of the observation slices in obs
        obs_u_view = @view obs[1:observable_minisections, i]
        obs_λ_view = @view obs[(observable_minisections + 1):(observable_minisections * 2), i]

        # Get views of the relevant minisection observations
        u_range = (last_minisection - observable_minisections + 1):last_minisection
        λ_range = (last_minisection - observable_minisections + 1):last_minisection

        # Copy and normalize directly into obs
        @. obs_u_view = minisection_observations_u[u_range] * max_pressure_inv
        @. obs_λ_view = minisection_observations_λ[λ_range]

        # Add the additional values
        obs[end - 2, i] = shocks * max_shocks_inv
        obs[end - 1, i] = target_shock_count * max_shocks_inv
        obs[end, i] = span * max_pressure_inv
    end

    return obs
end

function get_observable_minisections(obs_strategy::MultiSectionObservation)
    n_sections = obs_strategy.n_sections
    L = obs_strategy.L
    v = obs_strategy.look_ahead_speed
    dt = obs_strategy.dt
    m = obs_strategy.minisections_per_section


    observable_minisections = Int(floor(m * (1 + n_sections * v * dt / L)))
    return observable_minisections
end

function get_minisection_observations!(output, data, minisection_size)
    minisection_data = reshape(data, minisection_size, :)
    RDE.turbo_column_maximum!(output, minisection_data)
    return output
end

function get_minisection_observations(data, minisection_size)
    minisection_u = reshape(data, minisection_size, :)
    minisection_observations = RDE.turbo_column_maximum(minisection_u)
    return minisection_observations
end

function get_init_observation(obs_strategy::MultiSectionObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    obs_dim = get_observable_minisections(obs_strategy) * 2 + 3
    return Matrix{T}(undef, obs_dim, obs_strategy.n_sections)
end

@kwdef mutable struct CompositeObservation <: AbstractObservationStrategy
    fft_terms::Int = 8
    target_shock_count::Int = 4
end

function Base.show(io::IO, obs::CompositeObservation)
    return print(io, "CompositeObservation(fft_terms=$(obs.fft_terms))")
end

function Base.show(io::IO, ::MIME"text/plain", obs::CompositeObservation)
    println(io, "CompositeObservation:")
    println(io, "  fft_terms: $(obs.fft_terms)")
    return println(io, "  target_shock_count: $(obs.target_shock_count)")
end

function compute_observation!(obs, env::RDEEnv{T, A, O, R, V, OBS}, strategy::CompositeObservation) where {T, A, O, R, V, OBS}
    N = env.prob.params.N

    current_u = @view env.state[1:N]

    # env.cache.circ_u[:] .= current_u

    fft_u = abs.(fft(current_u)) #was circ_u

    n_terms = min(strategy.fft_terms, N ÷ 2 + 1)::Int

    # Create view into obs for FFT observations
    u_obs = @view obs[1:n_terms]

    # Take relevant FFT terms (view to avoid allocation)
    u_terms = @view fft_u[2:(n_terms + 1)] #the first is always one, so not useful
    u_terms_min = RDE.turbo_minimum(u_terms)
    u_terms_max = RDE.turbo_maximum(u_terms)

    # Normalize directly into obs
    u_range_inv = inv(u_terms_max - u_terms_min + T(1.0e-8))
    @. u_obs = (u_terms - u_terms_min) * u_range_inv

    dx = RDE.get_dx(env.prob)::T
    max_shocks = T(6)
    u_min, u_max = RDE.turbo_extrema(current_u)
    span = u_max - u_min

    # Write scalar values directly to the end of obs
    obs[n_terms + 1] = mean(env.prob.method.cache.u_p_current) / env.u_pmax
    obs[n_terms + 2] = RDE.count_shocks(current_u, dx) / max_shocks
    obs[n_terms + 3] = strategy.target_shock_count / max_shocks
    obs[n_terms + 4] = span / T(3)

    return obs
end

function get_init_observation(strategy::CompositeObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    return Vector{T}(undef, 2 * strategy.fft_terms + 5)
end


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

function compute_observation!(obs, env::RDEEnv{T, A, O, R, V, OBS}, obs_strategy::MultiCenteredObservation) where {T, A, O, R, V, OBS}
    n_sections = obs_strategy.n_sections
    minisections = obs_strategy.minisections
    minisections_per_section = minisections ÷ n_sections

    # Allocate temporary buffers for minisection observations
    minisection_observations_u = Vector{T}(undef, minisections)
    minisection_observations_λ = Vector{T}(undef, minisections)

    shocks, target_shock_count = compute_sectioned_observation!(
        minisection_observations_u, minisection_observations_λ, env, obs_strategy
    )

    # Fill the matrix directly
    for i in 1:n_sections
        # Get views into obs for this section
        obs_u_view = @view obs[1:minisections, i]
        obs_λ_view = @view obs[(minisections + 1):(minisections * 2), i]

        # Copy with circshift directly into obs
        obs_u_view .= circshift(minisection_observations_u, -minisections_per_section * (i - 1))
        obs_λ_view .= circshift(minisection_observations_λ, -minisections_per_section * (i - 1))

        # Add the additional values
        obs[end - 1, i] = shocks #is this needed?
        obs[end, i] = target_shock_count
    end

    return obs
end

function get_init_observation(obs_strategy::MultiCenteredObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    obs_dim = obs_strategy.minisections * 2 + 2
    return Matrix{T}(undef, obs_dim, obs_strategy.n_sections)
end

struct MeanInjectionPressureObservation <: AbstractObservationStrategy end

function compute_observation!(obs, env::RDEEnv{T, A, O, R, V, OBS}, strategy::MeanInjectionPressureObservation) where {T, A, O, R, V, OBS}
    obs[1] = mean(env.prob.method.cache.u_p_current)
    return obs
end

function get_init_observation(strategy::MeanInjectionPressureObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    return Vector{T}(undef, 1)
end
