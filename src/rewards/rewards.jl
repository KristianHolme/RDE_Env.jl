function set_reward!(env::AbstractRDEEnv, rew_strat::AbstractRewardStrategy)
    env.reward = compute_reward(env, rew_strat)
    return nothing
end

compute_reward(env::AbstractRDEEnv, rew_strat::AbstractRewardStrategy) =
    _compute_reward(env, rew_strat, env.cache.reward_cache)

struct WrappedRewardCache{IC <: AbstractCache} <: AbstractCache
    inner_cache::IC
end

function reset_cache!(cache::WrappedRewardCache)
    reset_cache!(cache.inner_cache)
    return nothing
end

struct USpanReward <: AbstractScalarRewardStrategy end

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, ::USpanReward, ::NoCache) where {T, A, O, R, G, V, OBS}
    N = env.prob.params.N
    u = @view env.state[1:N]
    u_min, u_max = RDE.turbo_extrema(u)
    return u_max - u_min
end

struct ScalarToVectorReward{T <: AbstractRewardStrategy} <: AbstractVectorRewardStrategy
    wrapped_reward::T
    n::Int
end

initialize_cache(rew_strat::ScalarToVectorReward, N::Int, ::Type{T}) where {T} =
    WrappedRewardCache(initialize_cache(rew_strat.wrapped_reward, N, T))

function _compute_reward(env::RDEEnv{T, A, O, R, G, V, OBS}, rew_strat::ScalarToVectorReward, cache::WrappedRewardCache) where {T, A, O, R, G, V, OBS}
    reward = _compute_reward(env, rew_strat.wrapped_reward, cache.inner_cache)
    return fill(reward, rew_strat.n)
end
