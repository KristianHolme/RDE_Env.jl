function compute_observation(env::RDEEnv{T, A, O, RW, V, OBS, M, RS, C}, rt::StateObservation) where {T, A, O, RW, V, OBS, M, RS, C}
    N = length(env.state) ÷ 2
    u = @view env.state[1:N]
    λ = @view env.state[(N + 1):end]

    ϵ = 1.0e-8
    u_max = max(RDE.turbo_maximum_abs(u), ϵ)
    λ_max = max(RDE.turbo_maximum_abs(λ), ϵ)

    normalized_state = similar(env.state)
    normalized_state[1:N] = u ./ u_max
    normalized_state[(N + 1):end] = λ ./ λ_max
    s_scaled = mean(env.prob.method.cache.s_current) / env.smax
    u_p_scaled = mean(env.prob.method.cache.u_p_current) / env.u_pmax
    return vcat(normalized_state, s_scaled, u_p_scaled)
end

function get_init_observation(::StateObservation, N::Int, ::Type{T}) where {T <: AbstractFloat}
    return Vector{T}(undef, 2N + 2)
end
