module RDE_EnvDRiLExt

using RDE_Env
using DRiL
export DRiLRDEEnv, DRiLMultiAgentRDEEnv

struct DRiLRDEEnv{T,A,O,R,V,OBS} <: DRiL.AbstractEnv
    core_env::RDEEnv{T,A,O,R,V,OBS}
    observation_space::DRiL.Box
    action_space::DRiL.Box
    function DRiLRDEEnv(core_env::RDEEnv{T,A,O,R,V,OBS}) where {T,A,O,R,V,OBS}
        observation_space = _observation_space(core_env, core_env.observation_strategy)
        action_space = _action_space(core_env, core_env.action_type)
        new{T,A,O,R,V,OBS}(core_env, observation_space, action_space)
    end
end

DRiL.reset!(env::DRiLRDEEnv) = _reset!(env.core_env)
DRiL.act!(env::DRiLRDEEnv{T,A,O,R,V,OBS}, action) where {T,A,O,R,V,OBS} = _act!(env.core_env, action)::V
DRiL.observe(env::DRiLRDEEnv{T,A,O,R,V,OBS}) where {T,A,O,R,V,OBS} = _observe(env.core_env)::OBS
DRiL.terminated(env::DRiLRDEEnv) = env.core_env.terminated
DRiL.truncated(env::DRiLRDEEnv) = env.core_env.truncated
DRiL.observation_space(env::DRiLRDEEnv) = env.observation_space
DRiL.action_space(env::DRiLRDEEnv) = env.action_space
DRiL.get_info(env::DRiLRDEEnv) = env.core_env.info



_action_space(::RDEEnv, ::ScalarPressureAction) = DRiL.Box([-1f0], [1f0])
_action_space(::RDEEnv, ::ScalarAreaScalarPressureAction) = DRiL.Box([-1f0, -1f0], [1f0, 1f0])
_action_space(::RDEEnv, action_type::VectorPressureAction) = DRiL.Box([-1f0 for _ in 1:action_type.n_sections], [1f0 for _ in 1:action_type.n_sections])

function _observation_space(core_env::RDEEnv, strategy::FourierObservation)
    N = core_env.prob.params.N
    n_fft_terms = min(strategy.fft_terms, N ÷ 2 + 1)
    total_terms = 2 * n_fft_terms + 2
    low = [[-1f0 for _ in 1:2*n_fft_terms]; 0f0; 0f0]
    high = [[1f0 for _ in 1:2*n_fft_terms]; 1f0; 1f0]
    return DRiL.Box(low, high)
end

function _observation_space(core_env::RDEEnv, ::StateObservation)
    N = length(core_env.state) ÷ 2
    low = [[-1f0 for _ in 1:N]; 0f0; 0f0]
    high = [[1f0 for _ in 1:N]; 1f0; 1f0]
    return DRiL.Box(low, high)
end

function _observation_space(core_env::RDEEnv, strategy::SectionedStateObservation)
    N = core_env.prob.params.N
    minisections = strategy.minisections
    # minisection_size = N ÷ minisections
    low = [[0f0 for _ in 1:2*minisections]; 0f0; 0f0]
    high = [[1f0 for _ in 1:2*minisections]; 1f0; 1f0]
    return DRiL.Box(low, high)
end

function _observation_space(core_env::RDEEnv, strategy::SampledStateObservation)
    N = core_env.prob.params.N
    n_samples = strategy.n_samples
    low = [[0f0 for _ in 1:2*n_samples]; 0f0]
    high = [[1f0 for _ in 1:2*n_samples]; 1f0]
    return DRiL.Box(low, high)
end

function _observation_space(core_env::RDEEnv, strategy::CompositeObservation)
    N = core_env.prob.params.N
    n_fft_terms = min(strategy.fft_terms, N ÷ 2 + 1)
    low = [[-1f0 for _ in 1:n_fft_terms]; 0f0; 0f0; 0f0]
    high = [[1f0 for _ in 1:n_fft_terms]; 1f0; 1f0; 1f0]
    return DRiL.Box(low, high)
end

function _observation_space(core_env::RDEEnv, ::MeanInjectionPressureObservation)
    return DRiL.Box([0f0], [Float32(core_env.u_pmax)])
end

function _observation_space(::RDEEnv, strategy::MultiCenteredObservation)
    obs_length = strategy.minisections * 2 + 2  # +2 for shocks, target_shock_count
    low = [0f0 for _ in 1:obs_length]
    high = [1f0 for _ in 1:obs_length]
    return DRiL.Box(low, high)
end

function _observation_space(::RDEEnv, strategy::MultiSectionObservation)
    observable_minisections = get_observable_minisections(strategy)
    obs_length = observable_minisections * 2 + 3  # +3 for shocks, target_shock_count, span
    low = [0f0 for _ in 1:obs_length]
    high = [1f0 for _ in 1:obs_length]
    return DRiL.Box(low, high)
end

function _multi_agent_action_space(::RDEEnv, action_type::VectorPressureAction)
    return DRiL.Box([-1f0], [1f0])
end



## multi-agent parallel env
struct DRiLMultiAgentRDEEnv{T,A,O,R,V,OBS} <: DRiL.AbstractParallelEnv
    core_env::RDEEnv{T,A,O,R,V,OBS}
    observation_space::DRiL.Box
    action_space::DRiL.Box
    n_envs::Int
    function DRiLMultiAgentRDEEnv(core_env::RDEEnv{T,A,O,R,V,OBS}) where {T,A,O,R,V,OBS}
        obs_strategy = core_env.observation_strategy
        @assert obs_strategy isa AbstractMultiAgentObservationStrategy
        @assert core_env.action_type isa VectorPressureAction
        action_type = core_env.action_type
        observation_space = _observation_space(core_env, obs_strategy)
        action_space = _multi_agent_action_space(core_env, action_type)
        n_envs = obs_strategy.n_sections
        @assert n_envs == action_type.n_sections
        new{T,A,O,R,V,OBS}(core_env, observation_space, action_space, n_envs)
    end
end

function DRiL.reset!(env::DRiLMultiAgentRDEEnv)
    _reset!(env.core_env)
end

function DRiL.observe(env::DRiLMultiAgentRDEEnv)
    observation_matrix = _observe(env.core_env)
    return eachslice(observation_matrix, dims=ndims(observation_matrix))
end

function DRiL.act!(env::DRiLMultiAgentRDEEnv{T,A,O,R,V,OBS}, actions::Vector{Vector{T}}) where {T,A,O,R,V,OBS}
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

DRiL.observation_space(env::DRiLMultiAgentRDEEnv) = env.observation_space
DRiL.action_space(env::DRiLMultiAgentRDEEnv) = env.action_space
DRiL.number_of_envs(env::DRiLMultiAgentRDEEnv) = env.n_envs
DRiL.terminated(env::DRiLMultiAgentRDEEnv) = fill(env.core_env.terminated, env.n_envs)
DRiL.truncated(env::DRiLMultiAgentRDEEnv) = fill(env.core_env.truncated, env.n_envs)

struct DRiLAgentPolicy <: AbstractRDEPolicy
    agent::AbstractAgent
    norm_env::Union{NormalizeWrapperEnv,Nothing}
end

function RDE_Env._predict_action(policy::DRiLAgentPolicy, observation::Vector)
    if !isnothing(policy.norm_env)
        DRiL.normalize_obs!(observation, policy.norm_env)
    end
    return predict_actions(policy.agent, [observation], deterministic=true)[1]
end

function RDE_Env._predict_action(policy::DRiLAgentPolicy, observation::Matrix)
    if !isnothing(policy.norm_env)
        DRiL.normalize_obs!(observation, policy.norm_env)
    end
    return predict_actions(policy.agent, eachcol(observation), deterministic=true)
end

function Base.show(io::IO, π::DRiLAgentPolicy)
    print(io, "DRiLAgentPolicy(agent=$(typeof(π.agent)))")
end

function Base.show(io::IO, ::MIME"text/plain", π::DRiLAgentPolicy)
    println(io, "DRiLAgentPolicy:")
    println(io, "  agent: $(typeof(π.agent))")
end

end #module