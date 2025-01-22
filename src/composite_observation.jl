@kwdef mutable struct CompositeObservation <: AbstractObservationStrategy
    fft_terms::Int = 8
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

    u_p_scaled = mean(env.prob.cache.u_p_current) / env.u_pmax
    dx = env.prob.x[2] - env.prob.x[1]
    max_shocks = 6f0
    normalized_shocks = RDE.count_shocks(current_u, dx)/max_shocks
    normalized_target_shock_count = env.reward_type.target_shock_count/max_shocks
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
