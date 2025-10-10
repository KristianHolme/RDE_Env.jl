function compute_observation(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, strategy::FourierObservation) where {T, A, O, RW, V, OBS, M, RS, C}
    N = env.prob.params.N

    current_u = @view env.state[1:N]
    current_λ = @view env.state[(N + 1):end]

    fft_u = abs.(fft(current_u))
    fft_λ = abs.(fft(current_λ))

    n_terms = min(strategy.fft_terms, N ÷ 2 + 1)

    u_terms = fft_u[1:n_terms]
    λ_terms = fft_λ[1:n_terms]

    u_terms_min = RDE.turbo_minimum(u_terms)
    u_terms_max = RDE.turbo_maximum(u_terms)
    λ_terms_min = RDE.turbo_minimum(λ_terms)
    λ_terms_max = RDE.turbo_maximum(λ_terms)
    u_obs = (u_terms .- u_terms_min) ./ (u_terms_max - u_terms_min + T(1.0e-8))
    λ_obs = (λ_terms .- λ_terms_min) ./ (λ_terms_max - λ_terms_min + T(1.0e-8))

    u_obs = 2 .* u_obs .- 1
    λ_obs = 2 .* λ_obs .- 1

    s_scaled = mean(env.prob.method.cache.s_current) / env.smax
    u_p_scaled = mean(env.prob.method.cache.u_p_current) / env.u_pmax

    return vcat(u_obs, λ_obs, s_scaled, u_p_scaled)
end

function get_init_observation(strategy::FourierObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    n_terms = min(strategy.fft_terms, N ÷ 2 + 1)
    return Vector{T}(undef, n_terms * 2 + 2)
end
