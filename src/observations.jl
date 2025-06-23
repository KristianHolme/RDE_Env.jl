"""
    compute_observation(env::AbstractRDEEnv, strategy::AbstractObservationStrategy)

Compute observation given an observation strategy.

# Arguments
- `env::AbstractRDEEnv`: RDE environment
- `strategy::AbstractObservationStrategy`: Observation strategy

# Returns
- Vector containing observation
"""
function compute_observation(env::AbstractRDEEnv, strategy::AbstractObservationStrategy)
    @error "compute_observation not implemented for strategy $(typeof(strategy))"
end

function compute_observation(env::RDEEnv{T}, strategy::FourierObservation) where T<:AbstractFloat
    N = env.prob.params.N
    
    current_u = @view env.state[1:N]
    current_λ = @view env.state[N+1:end]
    
    # env.cache.circ_u[:] .= current_u
    # env.cache.circ_λ[:] .= current_λ
    
    fft_u = abs.(fft(current_u)) #was circ_u
    fft_λ = abs.(fft(current_λ)) #was circ_λ
    
    n_terms = min(strategy.fft_terms, N ÷ 2 + 1)
    
    # Take relevant FFT terms
    u_terms = fft_u[1:n_terms]
    λ_terms = fft_λ[1:n_terms]
    
    # Min-max normalization to [0,1] range
    u_obs = (u_terms .- minimum(u_terms)) ./ (maximum(u_terms) - minimum(u_terms) + T(1e-8))
    λ_obs = (λ_terms .- minimum(λ_terms)) ./ (maximum(λ_terms) - minimum(λ_terms) + T(1e-8))
    
    # Scale to [-1,1] range if desired
    u_obs = 2 .* u_obs .- 1
    λ_obs = 2 .* λ_obs .- 1
    
    # Control parameters already naturally bounded
    s_scaled = mean(env.prob.method.cache.s_current) / env.smax
    u_p_scaled = mean(env.prob.method.cache.u_p_current) / env.u_pmax
    
    return vcat(u_obs, λ_obs, s_scaled, u_p_scaled)
end


function compute_observation(env::AbstractRDEEnv, rt::StateObservation)
    N = length(env.state) ÷ 2
    u = @view env.state[1:N]
    λ = @view env.state[N+1:end]
    
    ϵ = 1e-8
    u_max = max(maximum(abs.(u)), ϵ)
    λ_max = max(maximum(abs.(λ)), ϵ)
    
    normalized_state = similar(env.state)
    normalized_state[1:N] = u ./ u_max 
    normalized_state[N+1:end] = λ ./ λ_max
    s_scaled = mean(env.prob.method.cache.s_current) / env.smax
    u_p_scaled = mean(env.prob.method.cache.u_p_current) / env.u_pmax
    return vcat(normalized_state, s_scaled, u_p_scaled)
end

function compute_sectioned_observation(env::AbstractRDEEnv, obs_strategy::AbstractObservationStrategy)
    N = env.prob.params.N
    current_u = @view env.state[1:N]
    current_λ = @view env.state[N+1:end]
    # env.cache.circ_u[:] .= current_u
    # env.cache.circ_λ[:] .= current_λ
    # circ_u = env.cache.circ_u
    # circ_λ = env.cache.circ_λ
    max_shocks = 6f0
    max_pressure = 6f0

    dx = env.prob.x[2] - env.prob.x[1]
    minisections = obs_strategy.minisections
    minisection_size = N ÷ minisections

    minisection_observations_u = get_minisection_observations(current_u, minisection_size) ./ max_pressure
    minisection_observations_λ = get_minisection_observations(current_λ, minisection_size)


    shocks = RDE.count_shocks(current_u, dx) / max_shocks
    target_shock_count = obs_strategy.target_shock_count / max_shocks

    return minisection_observations_u, minisection_observations_λ, shocks, target_shock_count
end
function compute_observation(env::AbstractRDEEnv, strategy::SectionedStateObservation)
    (minisection_observations_u,
    minisection_observations_λ,
    shocks,
    target_shock_count) = compute_sectioned_observation(env, strategy)
    return vcat(minisection_observations_u, minisection_observations_λ, shocks, target_shock_count)
end

function compute_observation(env::AbstractRDEEnv, strategy::SampledStateObservation)
    N = env.prob.params.N
    n = strategy.n_samples
    
    u = @view env.state[1:N]
    λ = @view env.state[N+1:end]
    
    indices = round.(Int, range(1, N, length=n))
    sampled_u = u[indices]
    sampled_λ = λ[indices]
    
    normalized_time = env.t / env.prob.params.tmax
    ϵ = 1e-8
    u_max = max(maximum(abs.(sampled_u)), ϵ)
    λ_max = max(maximum(abs.(sampled_λ)), ϵ)
    
    sampled_u ./= u_max
    sampled_λ ./= λ_max
    return vcat(sampled_u, sampled_λ, normalized_time)
end

"""
    get_init_observation(strategy::AbstractObservationStrategy, N::Int)

Initialize observation vector for given strategy.

# Arguments
- `strategy`: Observation strategy
- `N`: Number of grid points

# Returns
Preallocated vector for observations
    """#TODO fix this, use another approach to get number of elements and then init vector
    function get_init_observation(strategy::FourierObservation, N::Int)
        n_terms = min(strategy.fft_terms, N ÷ 2 + 1)
        return Vector{Float32}(undef, n_terms * 2 + 2)
    end
    
    function get_init_observation(::StateObservation, N::Int)
    return Vector{Float32}(undef, 2N + 2)
end

function get_init_observation(strategy::SampledStateObservation, N::Int)
    return Vector{Float32}(undef, 2 * strategy.n_samples + 1)
end

function get_init_observation(strategy::SectionedStateObservation, N::Int)
    return Vector{Float32}(undef, 2 * strategy.minisections + 2)
end

@kwdef struct MultiSectionObservation <: AbstractMultiAgentObservationStrategy
    n_sections::Int = 4
    target_shock_count::Int = 3
    look_ahead_speed::Float32 = 1.65f0
    minisections_per_section::Int = 8
    dt::Float32 = 0.5f0
    L::Float32 = 2f0*π
end
MultiSectionObservation(n::Int) = MultiSectionObservation(n_sections=n)

function Base.show(io::IO, obs_strategy::MultiSectionObservation)
    if get(io, :compact, false)::Bool
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
    println(io, "  L: $(obs_strategy.L)")
end

function compute_observation(env::AbstractRDEEnv, obs_strategy::MultiSectionObservation)
    N = env.prob.params.N
    current_u = @view env.state[1:N]
    current_λ = @view env.state[N+1:end]
    # env.cache.circ_u[:] .= current_u
    # circ_u = env.cache.circ_u
    # circ_λ = env.cache.circ_λ
    max_shocks = 6f0
    max_pressure = 6f0

    n_sections = obs_strategy.n_sections
    dx = env.prob.x[2] - env.prob.x[1]
    m = obs_strategy.minisections_per_section
    section_size = N ÷ n_sections
    minisection_size = section_size ÷ m

    
    observable_minisections = get_observable_minisections(obs_strategy)

    minisection_observations_u = get_minisection_observations(current_u, minisection_size)
    minisection_observations_λ = get_minisection_observations(current_λ, minisection_size)
    last_minisection_in_sections = collect(m:m:m*n_sections)


    #concsiously not normalizing, leave it to PPO
    # observations = map(last_minisection_in_sections) do i
    #     @view minisection_observations[(i-observable_minisections+1):i]
    # end

    shocks = RDE.count_shocks(current_u, dx)
    target_shock_count = obs_strategy.target_shock_count
    span = maximum(current_u) - minimum(current_u)

    # Pre-allocate the final matrix
    obs_length = observable_minisections*2 + 3  # +3 for shocks, target_shock_count, span
    result = Matrix{Float32}(undef, obs_length, n_sections)
    
    # Fill the matrix directly
    for i in 1:n_sections
        # Copy the observation part
        last_minisection = last_minisection_in_sections[i]
        result[1:observable_minisections, i] .= minisection_observations_u[(last_minisection-observable_minisections+1):last_minisection] ./ max_pressure
        result[observable_minisections+1:observable_minisections*2, i] .= minisection_observations_λ[(last_minisection-observable_minisections+1):last_minisection]
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

function get_minisection_observations(data, minisection_size)
    minisection_u = reshape(data, minisection_size, :)
    minisection_observations = vec(maximum(minisection_u, dims=1))#TODO optimise??
    return minisection_observations
end

function get_init_observation(obs_strategy::MultiSectionObservation, N::Int)
    obs_dim = get_observable_minisections(obs_strategy)*2 + 3
    return Matrix{Float32}(undef, obs_dim, obs_strategy.n_sections)
end

@kwdef mutable struct CompositeObservation <: AbstractObservationStrategy
    fft_terms::Int = 8
    target_shock_count::Int = 4
end

function Base.show(io::IO, obs::CompositeObservation)
    print(io, "CompositeObservation(fft_terms=$(obs.fft_terms))")
end

function Base.show(io::IO, ::MIME"text/plain", obs::CompositeObservation)
    println(io, "CompositeObservation:")
    println(io, "  fft_terms: $(obs.fft_terms)")
    println(io, "  target_shock_count: $(obs.target_shock_count)")
end

function compute_observation(env::AbstractRDEEnv, strategy::CompositeObservation)
    N = env.prob.params.N
    
    current_u = @view env.state[1:N]
    
    # env.cache.circ_u[:] .= current_u
    
    fft_u = abs.(fft(current_u)) #was circ_u
    
    n_terms = min(strategy.fft_terms, N ÷ 2 + 1)
    
    # Take relevant FFT terms
    u_terms = fft_u[2:n_terms + 1] #the first is always one, so not useful
    u_obs = (u_terms .- minimum(u_terms)) ./ (maximum(u_terms) - minimum(u_terms) + 1f-8)

    u_p_scaled = mean(env.prob.method.cache.u_p_current) / env.u_pmax
    dx = env.prob.x[2] - env.prob.x[1]
    max_shocks = 6f0
    normalized_shocks = RDE.count_shocks(current_u, dx)/max_shocks
    normalized_target_shock_count = strategy.target_shock_count/max_shocks
    span = maximum(current_u) - minimum(current_u)
    normalized_span = span/3f0
    return vcat(u_obs, 
                u_p_scaled,
                normalized_shocks,
                normalized_target_shock_count,
                normalized_span)

end

function get_init_observation(strategy::CompositeObservation, N::Int)
    return Vector{Float32}(undef, 2 * strategy.fft_terms + 5)
end


@kwdef struct MultiCenteredObservation <: AbstractMultiAgentObservationStrategy
    n_sections::Int = 4
    target_shock_count::Int = 4
    minisections::Int = 32
end
MultiCenteredObservation(n::Int) = MultiCenteredObservation(n_sections=n)

function Base.show(io::IO, obs_strategy::MultiCenteredObservation)
    if get(io, :compact, false)::Bool
        print(io, "MultiCenteredObservation(n_sections=$(obs_strategy.n_sections))")
    else
        print(io, "MultiCenteredObservation(n_sections=$(obs_strategy.n_sections), target_shock_count=$(obs_strategy.target_shock_count))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", obs_strategy::MultiCenteredObservation)
    println(io, "MultiCenteredObservation:")
    println(io, "  n_sections: $(obs_strategy.n_sections)")
    println(io, "  target_shock_count: $(obs_strategy.target_shock_count)")
    println(io, "  minisections: $(obs_strategy.minisections)")
end

function compute_observation(env::AbstractRDEEnv, obs_strategy::MultiCenteredObservation)
    (minisection_observations_u,
    minisection_observations_λ,
    shocks,
    target_shock_count) = compute_sectioned_observation(env, obs_strategy)

    obs_length = obs_strategy.minisections*2 + 2  # +2 for shocks, target_shock_count
    n_sections = obs_strategy.n_sections
    minisections = obs_strategy.minisections
    minisections_per_section = minisections ÷ n_sections
    # Pre-allocate the final matrix
    result = Matrix{Float32}(undef, obs_length, n_sections)
    
    # Fill the matrix directly
    for i in 1:n_sections
        # Copy the observation part
        result[1:minisections, i] .= circshift(minisection_observations_u, -minisections_per_section*(i-1))
        result[minisections+1:minisections*2, i] .= circshift(minisection_observations_λ, -minisections_per_section*(i-1))
        # Add the additional values
        result[end-1, i] = shocks #is this needed?
        result[end, i] = target_shock_count
    end
    
    return result
end

function get_init_observation(obs_strategy::MultiCenteredObservation, N::Int)
    obs_dim = obs_strategy.minisections*2 + 2
    return Matrix{Float32}(undef, obs_dim, obs_strategy.n_sections)
end

struct MeanInjectionPressureObservation <: AbstractObservationStrategy end

function compute_observation(env::AbstractRDEEnv, strategy::MeanInjectionPressureObservation)
    return [mean(env.prob.method.cache.u_p_current)]
end

function get_init_observation(strategy::MeanInjectionPressureObservation, N::Int)
    return Vector{Float32}(undef, 1)
end