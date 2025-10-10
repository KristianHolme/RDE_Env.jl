function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::PeriodicityReward) where {T <: AbstractFloat, A, O, R, V, OBS}
    u, = RDE.split_sol_view(env.state)
    N::Int = env.prob.params.N
    dx::T = env.prob.x[2] - env.prob.x[1]
    L::T = env.prob.params.L
    shock_inds = RDE.shock_indices(u, dx)
    shocks::Int = length(shock_inds)

    cache = rt.cache
    if shocks > 1
        shift_steps::Int = N ÷ shocks
        errs::Vector{T} = zeros(T, shocks - 1)
        for i in 1:(shocks - 1)
            cache .= u
            circshift!(cache, u, -shift_steps * i)
            errs[i]::T = RDE.turbo_diff_norm(u, cache) / sqrt(N)
        end
        maxerr::T = RDE.turbo_maximum(errs)
        periodicity_reward = T(1) - (max(maxerr - T(0.08), zero(T)) / sqrt(T(3)))
    else
        periodicity_reward = T(1)
    end

    if shocks > 1
        optimal_spacing::T = L / shocks
        shock_spacing::Vector{T} = mod.(RDE.periodic_diff(shock_inds), N) .* dx
        shock_spacing_reward = T(1) - RDE.turbo_maximum_abs((shock_spacing .- optimal_spacing) ./ optimal_spacing)
    else
        shock_spacing_reward = T(1)
    end

    return (periodicity_reward::T + shock_spacing_reward::T) / T(2)
end

reward_value_type(::Type{T}, ::PeriodicityReward) where {T} = T

function Base.show(io::IO, rt::PeriodicityReward)
    return print(io, "PeriodicityReward(N=$(length(rt.cache)))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::PeriodicityReward)
    println(io, "PeriodicityReward:")
    println(io, "  cache size: $(length(rt.cache))")
    return nothing
end
