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

function compute_observation(env::AbstractRDEEnv{T}, strategy::FourierObservation) where T
    N = env.prob.params.N
    
    current_u = @view env.state[1:N]
    current_λ = @view env.state[N+1:end]
    
    env.cache.circ_u[:] .= current_u
    env.cache.circ_λ[:] .= current_λ
    
    fft_u = abs.(fft(env.cache.circ_u))
    fft_λ = abs.(fft(env.cache.circ_λ))
    
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

@kwdef struct MultiSectionObservation <: AbstractObservationStrategy
    n_sections::Int = 4
    target_shock_count::Int = 3
    look_ahead_speed::Float32 = 1.65f0
    minisections_per_section::Int = 8
    dt::Float32 = 0.5f0
    L::Float32 = 2f0*π
end
MultiSectionObservation(n::Int) = MultiSectionObservation(n_section=n)

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
    target_shock_count = obs_strategy.target_shock_count
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

@kwdef mutable struct CompositeObservation <: AbstractObservationStrategy
    fft_terms::Int = 8
    target_shock_count::Int = 4
end

function compute_observation(env::AbstractRDEEnv, strategy::CompositeObservation)
    N = env.prob.params.N
    
    current_u = @view env.state[1:N]
    
    env.cache.circ_u[:] .= current_u
    
    fft_u = abs.(fft(env.cache.circ_u))
    
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
