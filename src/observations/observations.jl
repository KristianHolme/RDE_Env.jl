# ============================================================================
# Cache Types
# ============================================================================

struct ObservationMinisectionCache{T <: AbstractFloat} <: AbstractCache
    minisection_u::Vector{T}
    minisection_λ::Vector{T}
end

struct ObservationPressureHistoryCache{T <: AbstractFloat} <: AbstractCache
    minisection_u::Vector{T}
    minisection_λ::Vector{T}
    pressure_history::Vector{T}
end

function reset_cache!(cache::ObservationPressureHistoryCache{T}) where {T}
    cache.pressure_history .= T(-1)
    return nothing
end

struct ObservationMultiCenteredPressureHistoryCache{T <: AbstractFloat} <: AbstractCache
    minisection_u::Vector{T}
    minisection_λ::Vector{T}
    pressure_history::Matrix{T}  # Each column is history for one section
end

function reset_cache!(cache::ObservationMultiCenteredPressureHistoryCache{T}) where {T}
    cache.pressure_history .= T(-1)
    return nothing
end

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------
function compute_sectioned_observation!(
        minisection_observations_u,
        minisection_observations_λ,
        env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C},
        obs_strategy::AbstractObservationStrategy
    ) where {T, A, O, RW, G, V, OBS, M, RS, C}
    prob = env.prob #::RDEProblem{T, M, RS, C}
    N = prob.params.N
    dx = RDE.get_dx(prob)::T
    control_shift_strategy = prob.control_shift_strategy

    if control_shift_strategy isa MovingFrameControlShift
        current_u = copy(env.state[1:N])
        current_λ = copy(env.state[(N + 1):end])
        shift = Int(round(RDE.get_control_shift(control_shift_strategy, current_u, env.t) / dx))
        if shift != 0
            circshift!(current_u, -shift)
            circshift!(current_λ, -shift)
        end
    else
        current_u = @view env.state[1:N]
        current_λ = @view env.state[(N + 1):end]
    end
    max_shocks = T(4)
    max_pressure = T(2)

    minisections = obs_strategy.minisections
    minisection_size = N ÷ minisections

    get_minisection_observations!(minisection_observations_u, current_u, minisection_size)
    @. minisection_observations_u /= max_pressure
    get_minisection_observations!(minisection_observations_λ, current_λ, minisection_size)

    shocks::T = RDE.count_shocks(current_u, dx) / max_shocks
    target_shock_count::T = get_target_shock_count(env) / max_shocks

    return shocks, target_shock_count
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
# ============================================================================
# Observation Types and Implementations
# ============================================================================

# ----------------------------------------------------------------------------
# FourierObservation
# ----------------------------------------------------------------------------

struct FourierObservation <: AbstractObservationStrategy
    fft_terms::Int
end

initialize_cache(::FourierObservation, N::Int, ::Type{T}) where {T} = NoCache()

#TODO fix this, use another approach to get number of elements and then init vector
function get_init_observation(strategy::FourierObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    n_terms = min(strategy.fft_terms, N ÷ 2 + 1)
    return Vector{T}(undef, n_terms * 2 + 2)
end
function compute_observation!(obs, env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, strategy::FourierObservation) where {T, A, O, RW, G, V, OBS, M, RS, C}
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
# ----------------------------------------------------------------------------
# StateObservation
# ----------------------------------------------------------------------------

struct StateObservation <: AbstractObservationStrategy end

initialize_cache(::StateObservation, N::Int, ::Type{T}) where {T} = NoCache()
function compute_observation!(obs, env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, rt::StateObservation) where {T, A, O, RW, G, V, OBS, M, RS, C}
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
function get_init_observation(::StateObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    return Vector{T}(undef, 2N + 2)
end
# ----------------------------------------------------------------------------
# SectionedStateObservation
# ----------------------------------------------------------------------------

@kwdef struct SectionedStateObservation <: AbstractObservationStrategy
    minisections::Int = 32
end

# ----------------------------------------------------------------------------
# SectionedStateWithPressureHistoryObservation
# ----------------------------------------------------------------------------

@kwdef struct SectionedStateWithPressureHistoryObservation <: AbstractObservationStrategy
    minisections::Int = 32
    history_length::Int = 14
end

initialize_cache(obs::SectionedStateObservation, N::Int, ::Type{T}) where {T} = begin
    minisection_size = N ÷ obs.minisections
    n_total_minisections = N ÷ minisection_size
    ObservationMinisectionCache{T}(zeros(T, n_total_minisections), zeros(T, n_total_minisections))
end

initialize_cache(obs::SectionedStateWithPressureHistoryObservation, N::Int, ::Type{T}) where {T} = begin
    minisection_size = N ÷ obs.minisections
    n_total_minisections = N ÷ minisection_size
    ObservationPressureHistoryCache{T}(
        zeros(T, n_total_minisections),
        zeros(T, n_total_minisections),
        fill(T(-1), obs.history_length)
    )
end

function compute_observation!(obs, env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, strategy::SectionedStateObservation) where {T, A, O, RW, G, V, OBS, M, RS, C}
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

function compute_observation!(obs, env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, strategy::SectionedStateWithPressureHistoryObservation) where {T, A, O, RW, G, V, OBS, M, RS, C}
    minisections = strategy.minisections
    history_length = strategy.history_length

    # Create views into obs for each component
    minisection_observations_u = @view obs[1:minisections]
    minisection_observations_λ = @view obs[(minisections + 1):(2 * minisections)]
    pressure_history_view = @view obs[(2 * minisections + 1):(2 * minisections + history_length)]

    shocks_normalized, target_shock_count_normalized = compute_sectioned_observation!(
        minisection_observations_u, minisection_observations_λ, env, strategy
    )

    # Get current mean injection pressure
    current_pressure = mean(env.prob.method.cache.u_p_current) / env.u_pmax

    # Get pressure history cache
    pressure_cache = env.cache.observation_cache

    # Check if this is the first observation (all values are -1)
    if all(≈(-1), pressure_cache.pressure_history)
        # First observation: fill entire history with current pressure
        pressure_cache.pressure_history .= current_pressure
        pressure_history_view .= current_pressure
    else
        # Subsequent observations: shift history and add new value
        circshift!(pressure_cache.pressure_history, -1)
        pressure_cache.pressure_history[end] = current_pressure
        pressure_history_view .= pressure_cache.pressure_history
    end

    # Write scalar values directly to the end of obs
    obs[2 * minisections + history_length + 1] = shocks_normalized
    obs[2 * minisections + history_length + 2] = target_shock_count_normalized

    return obs
end

# ----------------------------------------------------------------------------
# SampledStateObservation
# ----------------------------------------------------------------------------

struct SampledStateObservation <: AbstractObservationStrategy
    n_samples::Int
end

initialize_cache(::SampledStateObservation, N::Int, ::Type{T}) where {T} = NoCache()

function get_init_observation(strategy::SampledStateObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    return Vector{T}(undef, 2 * strategy.n_samples + 1)
end

function compute_observation!(obs, env::RDEEnv{T, A, O, R, G, V, OBS}, strategy::SampledStateObservation) where {T, A, O, R, G, V, OBS}
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
function get_init_observation(strategy::SectionedStateObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}

    return Vector{T}(undef, 2 * strategy.minisections + 2)
end

function get_init_observation(strategy::SectionedStateWithPressureHistoryObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    return Vector{T}(undef, 2 * strategy.minisections + strategy.history_length + 2)
end

# ----------------------------------------------------------------------------
# CompositeObservation
# ----------------------------------------------------------------------------

@kwdef mutable struct CompositeObservation <: AbstractObservationStrategy
    fft_terms::Int = 8
end

initialize_cache(::CompositeObservation, N::Int, ::Type{T}) where {T} = NoCache()

function Base.show(io::IO, obs::CompositeObservation)
    return print(io, "CompositeObservation(fft_terms=$(obs.fft_terms))")
end

function Base.show(io::IO, ::MIME"text/plain", obs::CompositeObservation)
    println(io, "CompositeObservation:")
    return println(io, "  fft_terms: $(obs.fft_terms)")
end

function compute_observation!(obs, env::RDEEnv{T, A, O, R, G, V, OBS}, strategy::CompositeObservation) where {T, A, O, R, G, V, OBS}
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
    obs[n_terms + 3] = get_target_shock_count(env) / max_shocks
    obs[n_terms + 4] = span / T(3)

    return obs
end

function get_init_observation(strategy::CompositeObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    return Vector{T}(undef, 2 * strategy.fft_terms + 5)
end

# ----------------------------------------------------------------------------
# MultiCenteredObservation
# ----------------------------------------------------------------------------

@kwdef struct MultiCenteredObservation <: AbstractMultiAgentObservationStrategy
    n_sections::Int = 4
    minisections::Int = 32
end
MultiCenteredObservation(n::Int) = MultiCenteredObservation(n_sections = n)

@kwdef struct MultiCenteredWithPressureHistoryObservation <: AbstractMultiAgentObservationStrategy
    n_sections::Int = 8
    minisections::Int = 32
    history_length::Int = 14
end
MultiCenteredWithPressureHistoryObservation(n::Int) = MultiCenteredWithPressureHistoryObservation(n_sections = n)

function Base.show(io::IO, obs_strategy::MultiCenteredObservation)
    return if get(io, :compact, false)::Bool
        print(io, "MultiCenteredObservation(n_sections=$(obs_strategy.n_sections))")
    else
        print(io, "MultiCenteredObservation(n_sections=$(obs_strategy.n_sections), minisections=$(obs_strategy.minisections))")
    end
end

initialize_cache(obs::MultiCenteredObservation, N::Int, ::Type{T}) where {T} = begin
    minisection_size = N ÷ obs.minisections
    n_total_minisections = N ÷ minisection_size
    ObservationMinisectionCache{T}(zeros(T, n_total_minisections), zeros(T, n_total_minisections))
end

initialize_cache(obs::MultiCenteredWithPressureHistoryObservation, N::Int, ::Type{T}) where {T} = begin
    ObservationMultiCenteredPressureHistoryCache{T}(
        zeros(T, obs.minisections),
        zeros(T, obs.minisections),
        fill(T(-1), obs.history_length, obs.n_sections)
    )
end

function Base.show(io::IO, ::MIME"text/plain", obs_strategy::MultiCenteredObservation)
    println(io, "MultiCenteredObservation:")
    println(io, "  n_sections: $(obs_strategy.n_sections)")
    return println(io, "  minisections: $(obs_strategy.minisections)")
end

function Base.show(io::IO, obs_strategy::MultiCenteredWithPressureHistoryObservation)
    return if get(io, :compact, false)::Bool
        print(io, "MultiCenteredWithPressureHistoryObservation(n_sections=$(obs_strategy.n_sections))")
    else
        print(io, "MultiCenteredWithPressureHistoryObservation(n_sections=$(obs_strategy.n_sections), minisections=$(obs_strategy.minisections), history_length=$(obs_strategy.history_length))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", obs_strategy::MultiCenteredWithPressureHistoryObservation)
    println(io, "MultiCenteredWithPressureHistoryObservation:")
    println(io, "  n_sections: $(obs_strategy.n_sections)")
    println(io, "  minisections: $(obs_strategy.minisections)")
    return println(io, "  history_length: $(obs_strategy.history_length)")
end

function compute_observation!(obs, env::RDEEnv{T, A, O, R, G, V, OBS}, obs_strategy::MultiCenteredObservation) where {T, A, O, R, G, V, OBS}
    n_sections = obs_strategy.n_sections
    minisections = obs_strategy.minisections
    minisections_per_section = minisections ÷ n_sections

    # Use cached buffers for minisection observations
    obs_cache = env.cache.observation_cache::ObservationMinisectionCache{T}
    minisection_observations_u = obs_cache.minisection_u
    minisection_observations_λ = obs_cache.minisection_λ

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

function compute_observation!(obs, env::RDEEnv{T, A, O, R, G, V, OBS}, obs_strategy::MultiCenteredWithPressureHistoryObservation) where {T, A, O, R, G, V, OBS}
    n_sections = obs_strategy.n_sections
    minisections = obs_strategy.minisections
    history_length = obs_strategy.history_length
    minisections_per_section = minisections ÷ n_sections

    # Use cached buffers
    obs_cache = env.cache.observation_cache::ObservationMultiCenteredPressureHistoryCache{T}
    minisection_observations_u = obs_cache.minisection_u
    minisection_observations_λ = obs_cache.minisection_λ
    pressure_history_cache = obs_cache.pressure_history

    shocks, target_shock_count = compute_sectioned_observation!(
        minisection_observations_u, minisection_observations_λ, env, obs_strategy
    )

    # Calculate points per section for extracting section-specific pressure
    N = env.prob.params.N
    points_per_section = N ÷ n_sections
    u_p_current = env.prob.method.cache.u_p_current

    # Fill the matrix for each section
    for i in 1:n_sections
        # Views into obs: [u, λ, pressure_history, shocks, target]
        obs_u_view = @view obs[1:minisections, i]
        obs_λ_view = @view obs[(minisections + 1):(minisections * 2), i]
        pressure_history_view = @view obs[(2 * minisections + 1):(2 * minisections + history_length), i]

        # Copy with circshift! for centered observations (in-place, no allocation)
        circshift!(obs_u_view, minisection_observations_u, -minisections_per_section * (i - 1))
        circshift!(obs_λ_view, minisection_observations_λ, -minisections_per_section * (i - 1))

        # Get section-specific mean pressure (normalized)
        section_start = (i - 1) * points_per_section + 1
        section_end = i * points_per_section
        section_u_p = @view u_p_current[section_start:section_end]
        current_section_pressure = mean(section_u_p) / env.u_pmax

        # Handle pressure history for this section
        section_history = @view pressure_history_cache[:, i]
        if all(≈(-1), section_history)
            # First observation: fill entire history with current pressure
            section_history .= current_section_pressure
            pressure_history_view .= current_section_pressure
        else
            # Subsequent observations: shift history in-place and add new value
            circshift!(section_history, -1)
            section_history[end] = current_section_pressure
            pressure_history_view .= section_history
        end

        # Add scalar values at the end
        obs[end - 1, i] = shocks
        obs[end, i] = target_shock_count
    end

    return obs
end

function get_init_observation(obs_strategy::MultiCenteredObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    obs_dim = obs_strategy.minisections * 2 + 2
    return Matrix{T}(undef, obs_dim, obs_strategy.n_sections)
end

function get_init_observation(obs_strategy::MultiCenteredWithPressureHistoryObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    obs_dim = obs_strategy.minisections * 2 + obs_strategy.history_length + 2
    return Matrix{T}(undef, obs_dim, obs_strategy.n_sections)
end

# ----------------------------------------------------------------------------
# MultiCenteredWithIndexObservation
# ----------------------------------------------------------------------------

@kwdef struct MultiCenteredWithIndexObservation <: AbstractMultiAgentObservationStrategy
    n_sections::Int = 4
    minisections::Int = 32
end
MultiCenteredWithIndexObservation(n::Int) = MultiCenteredWithIndexObservation(n_sections = n)

initialize_cache(obs::MultiCenteredWithIndexObservation, N::Int, ::Type{T}) where {T} = begin
    minisection_size = N ÷ obs.minisections
    n_total_minisections = N ÷ minisection_size
    ObservationMinisectionCache{T}(zeros(T, n_total_minisections), zeros(T, n_total_minisections))
end

function Base.show(io::IO, obs_strategy::MultiCenteredWithIndexObservation)
    return if get(io, :compact, false)::Bool
        print(io, "MultiCenteredWithIndexObservation(n_sections=$(obs_strategy.n_sections))")
    else
        print(io, "MultiCenteredWithIndexObservation(n_sections=$(obs_strategy.n_sections), minisections=$(obs_strategy.minisections))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", obs_strategy::MultiCenteredWithIndexObservation)
    println(io, "MultiCenteredWithIndexObservation:")
    println(io, "  n_sections: $(obs_strategy.n_sections)")
    return println(io, "  minisections: $(obs_strategy.minisections)")
end

function compute_observation!(obs, env::RDEEnv{T, A, O, R, G, V, OBS}, obs_strategy::MultiCenteredWithIndexObservation) where {T, A, O, R, G, V, OBS}
    n_sections = obs_strategy.n_sections
    minisections = obs_strategy.minisections
    minisections_per_section = minisections ÷ n_sections

    # Use cached buffers for minisection observations
    obs_cache = env.cache.observation_cache::ObservationMinisectionCache{T}
    minisection_observations_u = obs_cache.minisection_u
    minisection_observations_λ = obs_cache.minisection_λ

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
        obs[end - 2, i] = shocks
        obs[end - 1, i] = target_shock_count
        obs[end, i] = T(i) / T(n_sections)  # Section index normalized by number of sections
    end

    return obs
end

function get_init_observation(obs_strategy::MultiCenteredWithIndexObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    obs_dim = obs_strategy.minisections * 2 + 3  # +3 for shocks, target_shock_count, and section_index
    return Matrix{T}(undef, obs_dim, obs_strategy.n_sections)
end

# ----------------------------------------------------------------------------
# MeanInjectionPressureObservation
# ----------------------------------------------------------------------------

struct MeanInjectionPressureObservation <: AbstractObservationStrategy end

initialize_cache(::MeanInjectionPressureObservation, N::Int, ::Type{T}) where {T} = NoCache()

function compute_observation!(obs, env::RDEEnv{T, A, O, R, V, OBS}, strategy::MeanInjectionPressureObservation) where {T, A, O, R, V, OBS}
    obs[1] = mean(env.prob.method.cache.u_p_current)
    return obs
end

function get_init_observation(strategy::MeanInjectionPressureObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    return Vector{T}(undef, 1)
end

# ----------------------------------------------------------------------------
# MultiCenteredMovingFrameObservation
# ----------------------------------------------------------------------------

@kwdef struct MultiCenteredMovingFrameObservation <: AbstractMultiAgentObservationStrategy
    n_sections::Int = 4
    minisections::Int = 32
end
MultiCenteredMovingFrameObservation(n::Int) = MultiCenteredMovingFrameObservation(n_sections = n)

function Base.show(io::IO, obs_strategy::MultiCenteredMovingFrameObservation)
    return if get(io, :compact, false)::Bool
        print(io, "MultiCenteredMovingFrameObservation(n_sections=$(obs_strategy.n_sections))")
    else
        print(io, "MultiCenteredMovingFrameObservation(n_sections=$(obs_strategy.n_sections), minisections=$(obs_strategy.minisections))")
    end
end

initialize_cache(obs::MultiCenteredMovingFrameObservation, N::Int, ::Type{T}) where {T} = begin
    minisection_size = N ÷ obs.minisections
    n_total_minisections = N ÷ minisection_size
    ObservationMinisectionCache{T}(zeros(T, n_total_minisections), zeros(T, n_total_minisections))
end

function Base.show(io::IO, ::MIME"text/plain", obs_strategy::MultiCenteredMovingFrameObservation)
    println(io, "MultiCenteredMovingFrameObservation:")
    println(io, "  n_sections: $(obs_strategy.n_sections)")
    println(io, "  minisections: $(obs_strategy.minisections)")
    return nothing
end


function compute_observation!(obs, env::RDEEnv{T, A, O, R, G, V, OBS}, obs_strategy::MultiCenteredMovingFrameObservation) where {T, A, O, R, G, V, OBS}
    n_sections = obs_strategy.n_sections
    minisections = obs_strategy.minisections
    minisections_per_section = minisections ÷ n_sections

    # Use cached buffers for minisection observations
    obs_cache = env.cache.observation_cache::ObservationMinisectionCache{T}
    minisection_observations_u = obs_cache.minisection_u
    minisection_observations_λ = obs_cache.minisection_λ
    control_shift_strategy = env.prob.control_shift_strategy

    t = env.t
    dx = RDE.get_dx(env.prob)
    if t ≈ 0.0f0 #TODO: move into interface for control shift strategies?
        control_shift_strategy.position = 0.0f0
        control_shift_strategy.velocity = 0.0f0 # we dont shift at start
    else
        @assert !isnothing(env.prob.sol) "env.prob.sol is nothing"
        us, _ = RDE.split_sol(env.prob.sol.u)
        avg_speed = get_avg_wave_speed(us, env.prob.sol.t, dx)
        if avg_speed > zero(T)
            control_shift_strategy.velocity = avg_speed
        end
        control_shift_strategy.position += avg_speed * env.dt # position at end of step
        L = env.prob.params.L
        control_shift_strategy.position = mod(control_shift_strategy.position, L)
        control_shift_strategy.t_last = env.t
    end

    shocks, target_shock_count = compute_sectioned_observation!(
        minisection_observations_u, minisection_observations_λ, env, obs_strategy
    )

    if t ≈ 0.0f0
        # we guess at speed to shift u_p by in the first step
        control_shift_strategy.velocity = 1.8f0
    end

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

function get_init_observation(obs_strategy::MultiCenteredMovingFrameObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    obs_dim = obs_strategy.minisections * 2 + 2
    return Matrix{T}(undef, obs_dim, obs_strategy.n_sections)
end
