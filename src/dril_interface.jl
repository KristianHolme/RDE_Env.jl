DRiL.reset!(env::AbstractRDEEnv) = _reset!(env)
DRiL.act!(env::AbstractRDEEnv, action) = _act!(env, action)
DRiL.observe(env::AbstractRDEEnv) = _observe(env)
DRiL.terminated(env::AbstractRDEEnv) = env.terminated
DRiL.truncated(env::AbstractRDEEnv) = env.truncated
DRiL.get_info(env::AbstractRDEEnv) = env.info
DRiL.action_space(env::AbstractRDEEnv) = _action_space(env, env.action_strat)
DRiL.observation_space(env::AbstractRDEEnv) = _observation_space(env.prob.params, env.observation_strat)


"""
    _action_space(env::AbstractRDEEnv, action_strat::AbstractActionStrategy)

Get the action space for the environment and the action strategy.
"""
function _action_space end

function _action_space(env::RDEEnv, ::DirectScalarPressureAction)
    return DRiL.Box([0.0f0], [Float32(env.u_pmax)])
end
function _action_space(env::RDEEnv, action_strat::DirectVectorPressureAction)
    return DRiL.Box(
        [0.0f0 for _ in 1:action_strat.n_sections],
        [Float32(env.u_pmax) for _ in 1:action_strat.n_sections]
    )
end

function _multi_agent_observation_space(obs_space::DRiL.Box)
    return DRiL.Box(obs_space.low[:, 1], obs_space.high[:, 1])
end

function _multi_agent_action_space(params::RDEParam, ::DirectVectorPressureAction)
    return DRiL.Box([0.0f0], [Float32(params.u_pmax)])
end


"""
    MultiAgentRDEEnv <: DRiL.AbstractParallelEnv

Multi-agent wrapper for the RDEEnv. Implements the DRiL.AbstractParallelEnv interface.
"""
struct MultiAgentRDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C} <: DRiL.AbstractParallelEnv
    core_env::RDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C}
    observation_space::DRiL.Box
    action_space::DRiL.Box
    n_envs::Int
    function MultiAgentRDEEnv(core_env::RDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C}) where {T, A, O, RW, CS, V, OBS, M, RS, C}
        obs_strategy = core_env.observation_strat
        @assert obs_strategy isa AbstractMultiAgentObservationStrategy
        @assert core_env.action_strat isa AbstractVectorActionStrategy
        action_strat = core_env.action_strat
        observation_space = _multi_agent_observation_space(
            _observation_space(core_env.prob.params, obs_strategy)
        )
        action_space = _multi_agent_action_space(core_env.prob.params, action_strat)
        n_envs = obs_strategy.n_sections
        @assert n_envs == action_strat.n_sections
        return new{T, A, O, RW, CS, V, OBS, M, RS, C}(core_env, observation_space, action_space, n_envs)
    end
end

function DRiL.reset!(env::MultiAgentRDEEnv)
    return _reset!(env.core_env)
end

function DRiL.observe(env::MultiAgentRDEEnv)
    observation_matrix = _observe(env.core_env)
    return collect.(eachslice(observation_matrix, dims = ndims(observation_matrix)))
end

function DRiL.act!(env::MultiAgentRDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C}, actions::AbstractVector{<:AbstractVector}) where {T, A, O, RW, CS, V, OBS, M, RS, C}
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

DRiL.observation_space(env::MultiAgentRDEEnv) = env.observation_space
DRiL.action_space(env::MultiAgentRDEEnv) = env.action_space
DRiL.number_of_envs(env::MultiAgentRDEEnv) = env.n_envs
DRiL.terminated(env::MultiAgentRDEEnv) = fill(env.core_env.terminated, env.n_envs)
DRiL.truncated(env::MultiAgentRDEEnv) = fill(env.core_env.truncated, env.n_envs)
DRiL.get_info(env::MultiAgentRDEEnv) = [copy(env.core_env.info) for _ in 1:env.n_envs]
