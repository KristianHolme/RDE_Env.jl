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
    s_scaled = mean(env.prob.cache.s_current) / env.smax
    u_p_scaled = mean(env.prob.cache.u_p_current) / env.u_pmax
    
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
    s_scaled = mean(env.prob.cache.s_current) / env.smax
    u_p_scaled = mean(env.prob.cache.u_p_current) / env.u_pmax
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