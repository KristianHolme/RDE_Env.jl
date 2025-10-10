function compute_observation(env::RDEEnv{T, A, O, R, V, OBS}, strategy::SampledStateObservation) where {T, A, O, R, V, OBS}
    N = env.prob.params.N
    n = strategy.n_samples

    u = @view env.state[1:N]
    λ = @view env.state[(N + 1):end]

    indices = round.(Int, range(1, N, length = n))
    sampled_u = u[indices]
    sampled_λ = λ[indices]

    normalized_time = env.t / env.prob.params.tmax
    ϵ = 1.0e-8
    u_max = max(RDE.turbo_maximum_abs(sampled_u), ϵ)
    λ_max = max(RDE.turbo_maximum_abs(sampled_λ), ϵ)

    sampled_u ./= u_max
    sampled_λ ./= λ_max
    return vcat(sampled_u, sampled_λ, normalized_time)
end

function get_init_observation(strategy::SampledStateObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    return Vector{T}(undef, 2 * strategy.n_samples + 1)
end
