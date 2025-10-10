struct MultiplicativeReward <: AbstractRDEReward
    rewards::Vector{AbstractRDEReward}

    function MultiplicativeReward(rewards::Vector{AbstractRDEReward})
        return new(rewards)
    end

    function MultiplicativeReward(rewards::AbstractRDEReward...)
        return new([rewards...])
    end
end

function Base.show(io::IO, rt::MultiplicativeReward)
    return print(io, "MultiplicativeReward($(length(rt.rewards)) components)")
end

function Base.show(io::IO, ::MIME"text/plain", rt::MultiplicativeReward)
    println(io, "MultiplicativeReward:")
    println(io, "  components: $(length(rt.rewards))")
    for (i, reward) in enumerate(rt.rewards)
        println(io, "  reward $i: $(typeof(reward))")
    end
    return
end

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::MultiplicativeReward) where {T, A, O, R, V, OBS}
    reward = compute_reward(env, rt.rewards[1])
    for r in rt.rewards[2:end]
        next_reward = compute_reward(env, r)
        reward = next_reward .* reward
    end
    return reward
end

function reward_value_from_wrapper(::Type{T}, rts::Vector{<:AbstractRDEReward}) where {T}
    value_types = reward_value_type.(T, rts)
    if any(value_types .== Vector{T})
        return Vector{T}
    else
        return T
    end
end

function reward_value_type(::Type{T}, rt::MultiplicativeReward) where {T}
    return reward_value_from_wrapper(T, rt.rewards)
end

function reset_reward!(rt::MultiplicativeReward)
    for r in rt.rewards
        reset_reward!(r)
    end
    return nothing
end

mutable struct ExponentialAverageReward{T <: AbstractRDEReward, U <: AbstractFloat} <: AbstractRDEReward
    wrapped_reward::T
    α::U
    average::Union{Nothing, U, Vector{U}}

    function ExponentialAverageReward{T, U}(wrapped_reward::T; α::U = U(0.2)) where {T <: AbstractRDEReward, U <: AbstractFloat}
        return new{T, U}(wrapped_reward, α, nothing)
    end
    function ExponentialAverageReward(wrapped_reward::T; α::Float32 = 0.2f0) where {T <: AbstractRDEReward}
        return ExponentialAverageReward{T, Float32}(wrapped_reward, α = α)
    end
end

function Base.show(io::IO, rt::ExponentialAverageReward)
    return print(io, "ExponentialAverageReward(α=$(rt.α), wrapped=$(typeof(rt.wrapped_reward)))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::ExponentialAverageReward)
    println(io, "ExponentialAverageReward:")
    println(io, "  α: $(rt.α)")
    println(io, "  wrapped_reward: $(typeof(rt.wrapped_reward))")
    return println(io, "  current_average: $(rt.average)")
end

function reward_value_type(::Type{T}, rt::ExponentialAverageReward) where {T}
    return reward_value_type(T, rt.wrapped_reward)
end

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::ExponentialAverageReward) where {T, A, O, R, V, OBS}
    current_reward = compute_reward(env, rt.wrapped_reward)

    if isnothing(rt.average)
        rt.average = current_reward
        return current_reward
    else
        rt.average = rt.α .* current_reward .+ (1 - rt.α) .* rt.average
        return rt.average
    end
end

function reset_reward!(rt::ExponentialAverageReward)
    rt.average = nothing
    reset_reward!(rt.wrapped_reward)
    return nothing
end

mutable struct TransitionBasedReward{T <: AbstractRDEReward, U <: AbstractFloat} <: AbstractRDEReward
    wrapped_reward::T
    target_shocks::Int
    reward_stability_length::U
    reward_threshold::U

    past_rewards::Vector{U}
    past_shock_counts::Vector{Int}
    transition_found::Bool

    function TransitionBasedReward{T, U}(wrapped_reward::T; target_shocks::Int = 3,
            reward_stability_length::U = U(20), reward_threshold::U = U(0.99)) where {T <: AbstractRDEReward, U <: AbstractFloat}
        return new{T, U}(wrapped_reward, target_shocks, reward_stability_length, reward_threshold, U[], Int[], false)
    end
    function TransitionBasedReward(wrapped_reward::T; target_shocks::Int = 3,
            reward_stability_length::Float32 = 20.0f0, reward_threshold::Float32 = 0.99f0) where {T <: AbstractRDEReward}
        return TransitionBasedReward{T, Float32}(wrapped_reward, target_shocks = target_shocks, reward_stability_length = reward_stability_length, reward_threshold = reward_threshold)
    end
end

function Base.show(io::IO, rt::TransitionBasedReward)
    return print(io, "TransitionBasedReward(target_shocks=$(rt.target_shocks), wrapped=$(typeof(rt.wrapped_reward)))")
end

function Base.show(io::IO, ::MIME"text/plain", rt::TransitionBasedReward)
    println(io, "TransitionBasedReward:")
    println(io, "  target_shocks: $(rt.target_shocks)")
    println(io, "  reward_stability_length: $(rt.reward_stability_length)")
    println(io, "  reward_threshold: $(rt.reward_threshold)")
    println(io, "  wrapped_reward: $(typeof(rt.wrapped_reward))")
    println(io, "  transition_found: $(rt.transition_found)")
    return println(io, "  history_length: $(length(rt.past_rewards))")
end

reward_value_type(::Type{T}, ::TransitionBasedReward) where {T} = T

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::TransitionBasedReward{S, T}) where {T, A, O, R, V, OBS, S}
    wrapped_reward_value = compute_reward(env, rt.wrapped_reward)

    reward_scalar = if wrapped_reward_value isa AbstractVector
        minimum(wrapped_reward_value)
    else
        wrapped_reward_value
    end

    N = env.prob.params.N
    u = @view env.state[1:N]
    dx = env.prob.x[2] - env.prob.x[1]
    current_shock_count = RDE.count_shocks(u, dx)

    push!(rt.past_rewards, T(reward_scalar))
    push!(rt.past_shock_counts, current_shock_count)

    if length(rt.past_shock_counts) >= 2
        rt.transition_found = detect_transition_realtime(rt, env.dt)

        if rt.transition_found
            env.terminated = true
            @logmsg LogLevel(-500) "Transition detected! Terminating environment at t=$(env.t)"
            env.info["Termination.Reason"] = "Transition detected"
            env.info["Termination.env_t"] = env.t
            return T(100.0)
        end
    end

    return T(-env.dt)
end

function detect_transition_realtime(rt::TransitionBasedReward{T, U}, dt::U) where {T, U <: AbstractFloat}
    stability_steps = round(Int, rt.reward_stability_length / dt)
    if length(rt.past_shock_counts) < stability_steps
        @debug "Not enough shock counts to detect transition"
        return false
    end

    stability_rewards = rt.past_rewards[(end - stability_steps + 1):end]
    stability_shock_counts = rt.past_shock_counts[(end - stability_steps + 1):end]
    if all(stability_shock_counts .== rt.target_shocks) && minimum(stability_rewards) > rt.reward_threshold
        return true
    end
    return false
end

function reset_reward!(rt::TransitionBasedReward)
    empty!(rt.past_rewards)
    empty!(rt.past_shock_counts)
    rt.transition_found = false
    reset_reward!(rt.wrapped_reward)
    return nothing
end

struct ScalarToVectorReward{T <: AbstractRDEReward} <: AbstractRDEReward
    wrapped_reward::T
    n::Int
end

reward_value_type(::Type{T}, ::ScalarToVectorReward) where {T} = Vector{T}

function compute_reward(env::RDEEnv{T, A, O, R, V, OBS}, rt::ScalarToVectorReward) where {T, A, O, R, V, OBS}
    reward = compute_reward(env, rt.wrapped_reward)
    return fill(reward, rt.n)
end

function reset_reward!(rt::ScalarToVectorReward)
    return reset_reward!(rt.wrapped_reward)
end
