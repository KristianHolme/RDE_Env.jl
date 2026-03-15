DrillInterface.reset!(env::AbstractRDEEnv) = _reset!(env)
DrillInterface.act!(env::AbstractRDEEnv, action) = _act!(env, action)
DrillInterface.observe(env::AbstractRDEEnv) = _observe(env)
DrillInterface.terminated(env::AbstractRDEEnv) = env.terminated
DrillInterface.truncated(env::AbstractRDEEnv) = env.truncated
DrillInterface.get_info(env::AbstractRDEEnv) = env.info
DrillInterface.action_space(env::AbstractRDEEnv) = _action_space(env, env.action_strat)
DrillInterface.observation_space(env::AbstractRDEEnv) = _observation_space(env.prob.params, env.observation_strat)


"""
    _action_space(env::AbstractRDEEnv, action_strat::AbstractActionStrategy)

Get the action space for the environment and the action strategy.
"""
function _action_space end

function _action_space(env::RDEEnv, ::DirectScalarPressureAction)
    return DrillInterface.Box([0.0f0], [Float32(env.u_pmax)])
end
function _action_space(env::RDEEnv, action_strat::DirectVectorPressureAction)
    return DrillInterface.Box(
        [0.0f0 for _ in 1:action_strat.n_sections],
        [Float32(env.u_pmax) for _ in 1:action_strat.n_sections]
    )
end

function _multi_agent_observation_space(obs_space::DrillInterface.Box)
    return DrillInterface.Box(obs_space.low[:, 1], obs_space.high[:, 1])
end

function _multi_agent_action_space(env::RDEEnv, ::DirectVectorPressureAction)
    return DrillInterface.Box([0.0f0], [Float32(env.u_pmax)])
end


"""
    MultiAgentRDEEnv <: DrillInterface.AbstractParallelEnv

Multi-agent wrapper for the RDEEnv. Implements the DrillInterface.AbstractParallelEnv interface.
"""
struct MultiAgentRDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C} <: DrillInterface.AbstractParallelEnv
    core_env::RDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C}
    observation_space::DrillInterface.Box #TODO abstract type?
    action_space::DrillInterface.Box
    n_envs::Int
    function MultiAgentRDEEnv(core_env::RDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C}) where {T, A, O, RW, CS, V, OBS, M, RS, C}
        obs_strategy = core_env.observation_strat
        @assert obs_strategy isa AbstractMultiAgentObservationStrategy
        @assert core_env.action_strat isa AbstractVectorActionStrategy
        action_strat = core_env.action_strat
        observation_space = _multi_agent_observation_space(
            _observation_space(core_env.prob.params, obs_strategy)
        )
        action_space = _multi_agent_action_space(core_env, action_strat)
        n_envs = obs_strategy.n_sections
        @assert n_envs == action_strat.n_sections
        return new{T, A, O, RW, CS, V, OBS, M, RS, C}(core_env, observation_space, action_space, n_envs)
    end
end

function DrillInterface.reset!(env::MultiAgentRDEEnv)
    return _reset!(env.core_env)
end

function DrillInterface.observe(env::MultiAgentRDEEnv)
    observation_matrix = _observe(env.core_env)
    return collect.(eachslice(observation_matrix, dims = ndims(observation_matrix)))
end

function DrillInterface.act!(env::MultiAgentRDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C}, actions::AbstractVector{<:AbstractVector}) where {T, A, O, RW, CS, V, OBS, M, RS, C}
    combined_action = vcat(actions...)::Vector{T}
    rewards = _act!(env.core_env, combined_action)::V
    info = env.core_env.info
    infos = [copy(info) for _ in 1:env.n_envs]
    if env.core_env.truncated
        observations = _observe(env.core_env)
        for i in 1:env.n_envs
            infos[i]["terminal_observation"] = observations[:, i]
        end
    end
    terminateds = [env.core_env.terminated for _ in 1:env.n_envs]
    truncateds = [env.core_env.truncated for _ in 1:env.n_envs]
    if env.core_env.terminated || env.core_env.truncated
        _reset!(env.core_env)
    end
    return rewards, terminateds, truncateds, infos
end

Random.seed!(env::MultiAgentRDEEnv, seed::Int) = Random.seed!(env.core_env, seed)

DrillInterface.observation_space(env::MultiAgentRDEEnv) = env.observation_space
DrillInterface.action_space(env::MultiAgentRDEEnv) = env.action_space
DrillInterface.number_of_envs(env::MultiAgentRDEEnv) = env.n_envs
DrillInterface.terminated(env::MultiAgentRDEEnv) = fill(env.core_env.terminated, env.n_envs)
DrillInterface.truncated(env::MultiAgentRDEEnv) = fill(env.core_env.truncated, env.n_envs)
DrillInterface.get_info(env::MultiAgentRDEEnv) = [copy(env.core_env.info) for _ in 1:env.n_envs]
