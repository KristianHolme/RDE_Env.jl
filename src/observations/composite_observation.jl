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

function compute_observation(env::RDEEnv{T, A, O, R, V, OBS}, strategy::CompositeObservation) where {T, A, O, R, V, OBS}
    N = env.prob.params.N

    current_u = @view env.state[1:N]
    fft_u = abs.(fft(current_u))

    n_terms = min(strategy.fft_terms, N ÷ 2 + 1)::Int
    u_terms = fft_u[2:(n_terms + 1)]
    u_terms_min = RDE.turbo_minimum(u_terms)
    u_terms_max = RDE.turbo_maximum(u_terms)
    u_obs = (u_terms .- u_terms_min) ./ (u_terms_max - u_terms_min + T(1.0e-8))

    u_p_scaled = mean(env.prob.method.cache.u_p_current) / env.u_pmax
    dx = RDE.get_dx(env.prob)::T
    max_shocks = T(6)
    normalized_shocks = RDE.count_shocks(current_u, dx) / max_shocks
    normalized_target_shock_count = strategy.target_shock_count / max_shocks
    u_min, u_max = RDE.turbo_extrema(current_u)
    span = u_max - u_min
    normalized_span = span / T(3)
    return vcat(
        u_obs,
        u_p_scaled,
        normalized_shocks,
        normalized_target_shock_count,
        normalized_span
    )
end

function get_init_observation(strategy::CompositeObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    return Vector{T}(undef, 2 * strategy.fft_terms + 5)
end
