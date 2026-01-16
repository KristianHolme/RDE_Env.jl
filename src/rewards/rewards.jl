function set_reward!(env::AbstractRDEEnv, rew_strat::AbstractRewardStrategy, context::AbstractCache)
    env.reward = compute_reward(env, rew_strat, env.cache.reward_cache, context)
    return env
end

"""
    compute_reward(env::RDEEnv, rew_strat::AbstractRewardStrategy, cache::AbstractCache, context::AbstractCache)

Compute the reward from the environment.
"""
function compute_reward end

struct WrappedRewardCache{IC <: AbstractCache} <: AbstractCache
    inner_cache::IC
end

function reset_cache!(cache::WrappedRewardCache)
    reset_cache!(cache.inner_cache)
    return nothing
end

struct USpanReward <: AbstractScalarRewardStrategy end

function compute_reward(env::RDEEnv{T, A, O, R, CS, V, OBS}, ::USpanReward, ::NoCache, ::AbstractCache) where {T, A, O, R, CS, V, OBS}
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

function compute_reward(env::RDEEnv{T, A, O, R, CS, V, OBS}, rew_strat::ScalarToVectorReward, cache::WrappedRewardCache, context::AbstractCache) where {T, A, O, R, CS, V, OBS}
    reward = compute_reward(env, rew_strat.wrapped_reward, cache.inner_cache, context)
    return fill(reward, rew_strat.n)
end
