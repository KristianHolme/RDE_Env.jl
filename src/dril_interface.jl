DRiL.reset!(env::AbstractRDEEnv) = _reset!(env)
DRiL.act!(env::AbstractRDEEnv, action) = _act!(env, action)
DRiL.observe(env::AbstractRDEEnv) = _observe(env)
DRiL.terminated(env::AbstractRDEEnv) = env.terminated
DRiL.truncated(env::AbstractRDEEnv) = env.truncated
DRiL.get_info(env::AbstractRDEEnv) = env.info
DRiL.action_space(env::AbstractRDEEnv) = _action_space(env, env.action_strat)
DRiL.observation_space(env::AbstractRDEEnv) = _observation_space(env, env.observation_strat)

function _action_space(::RDEEnv, ::ScalarPressureAction)
    return DRiL.Box([-1.0f0], [1.0f0])
end
function _action_space(::RDEEnv, ::ScalarAreaScalarPressureAction)
    return DRiL.Box([-1.0f0, -1.0f0], [1.0f0, 1.0f0])
end
function _action_space(::RDEEnv, action_strat::VectorPressureAction)
    return DRiL.Box(
        [-1.0f0 for _ in 1:action_strat.n_sections],
        [1.0f0 for _ in 1:action_strat.n_sections]
    )
end
function _action_space(::RDEEnv, ::PIDAction)
    return DRiL.Box([-1.0f0, -1.0f0, -1.0f0], [1.0f0, 1.0f0, 1.0f0])
end
function _action_space(::RDEEnv, ::LinearScalarPressureAction)
    return DRiL.Box([-1.0f0], [1.0f0])
end
function _action_space(env::RDEEnv, ::DirectScalarPressureAction)
    return DRiL.Box([0.0f0], [Float32(env.u_pmax)])
end
function _action_space(::RDEEnv, action_strat::LinearVectorPressureAction)
    return DRiL.Box(
        [-1.0f0 for _ in 1:action_strat.n_sections],
        [1.0f0 for _ in 1:action_strat.n_sections]
    )
end
function _action_space(env::RDEEnv, action_strat::DirectVectorPressureAction)
    return DRiL.Box(
        [0.0f0 for _ in 1:action_strat.n_sections],
        [Float32(env.u_pmax) for _ in 1:action_strat.n_sections]
    )
end
function _action_space(::RDEEnv, ::LinearScalarPressureWithDtAction)
    return DRiL.Box([-1.0f0, -1.0f0], [1.0f0, 1.0f0])
end

function _observation_space(core_env::RDEEnv, strategy::FourierObservation)
    N = core_env.prob.params.N
    n_fft_terms = min(strategy.fft_terms, N ÷ 2 + 1)
    total_terms = 2 * n_fft_terms + 2
    low = [[-1.0f0 for _ in 1:(2 * n_fft_terms)]; 0.0f0; 0.0f0]
    high = [[1.0f0 for _ in 1:(2 * n_fft_terms)]; 1.0f0; 1.0f0]
    return DRiL.Box(low, high)
end

function _observation_space(core_env::RDEEnv, ::StateObservation)
    N = length(core_env.state) ÷ 2
    low = [[-1.0f0 for _ in 1:N]; 0.0f0; 0.0f0]
    high = [[1.0f0 for _ in 1:N]; 1.0f0; 1.0f0]
    return DRiL.Box(low, high)
end

function _observation_space(core_env::RDEEnv, ::FullStateObservation)
    N = length(core_env.state) ÷ 2
    bound = 1.0e6f0
    low = [fill(-bound, N); fill(-bound, N)]
    high = [fill(bound, N); fill(bound, N)]
    return DRiL.Box(low, high)
end

function _observation_space(core_env::RDEEnv, strategy::SectionedStateObservation)
    minisections = strategy.minisections
    low = [[0.0f0 for _ in 1:(2 * minisections)]; 0.0f0; 0.0f0]
    high = [[1.0f0 for _ in 1:(2 * minisections)]; 1.0f0; 1.0f0]
    return DRiL.Box(low, high)
end

function _observation_space(core_env::RDEEnv, strategy::SectionedStateMovingFrameObservation)
    minisections = strategy.minisections
    low = [[0.0f0 for _ in 1:(2 * minisections)]; 0.0f0; 0.0f0]
    high = [[1.0f0 for _ in 1:(2 * minisections)]; 1.0f0; 1.0f0]
    return DRiL.Box(low, high)
end

function _observation_space(core_env::RDEEnv, strategy::SampledStateObservation)
    n_samples = strategy.n_samples
    low = [[0.0f0 for _ in 1:(2 * n_samples)]; 0.0f0]
    high = [[1.0f0 for _ in 1:(2 * n_samples)]; 1.0f0]
    return DRiL.Box(low, high)
end

function _observation_space(core_env::RDEEnv, strategy::CompositeObservation)
    N = core_env.prob.params.N
    n_fft_terms = min(strategy.fft_terms, N ÷ 2 + 1)
    low = [[-1.0f0 for _ in 1:n_fft_terms]; 0.0f0; 0.0f0; 0.0f0]
    high = [[1.0f0 for _ in 1:n_fft_terms]; 1.0f0; 1.0f0; 1.0f0]
    return DRiL.Box(low, high)
end

function _observation_space(core_env::RDEEnv, ::MeanInjectionPressureObservation)
    return DRiL.Box([0.0f0], [Float32(core_env.u_pmax)])
end

function _observation_space(::RDEEnv, strategy::MultiCenteredObservation)
    obs_length = strategy.minisections * 2 + 2
    low = [0.0f0 for _ in 1:obs_length]
    high = [1.0f0 for _ in 1:obs_length]
    return DRiL.Box(low, high)
end

function _observation_space(core_env::RDEEnv, strategy::SectionedStateWithPressureHistoryObservation)
    minisections = strategy.minisections
    history_length = strategy.history_length
    obs_length = 2 * minisections + history_length + 2
    low = [[0.0f0 for _ in 1:(2 * minisections + history_length)]; 0.0f0; 0.0f0]
    high = [[1.0f0 for _ in 1:(2 * minisections + history_length)]; 1.0f0; 1.0f0]
    return DRiL.Box(low, high)
end

function _observation_space(core_env::RDEEnv, strategy::MultiCenteredWithPressureHistoryObservation)
    minisections = strategy.minisections
    history_length = strategy.history_length
    obs_length = 2 * minisections + history_length + 2
    low = [[0.0f0 for _ in 1:(2 * minisections + history_length)]; 0.0f0; 0.0f0]
    high = [[1.0f0 for _ in 1:(2 * minisections + history_length)]; 1.0f0; 1.0f0]
    return DRiL.Box(low, high)
end

function _observation_space(core_env::RDEEnv, strategy::MultiCenteredMovingFrameObservation)
    minisections = strategy.minisections
    obs_length = 2 * minisections + 2
    low = [0.0f0 for _ in 1:obs_length]
    high = [1.0f0 for _ in 1:obs_length]
    return DRiL.Box(low, high)
end

function _observation_space(core_env::RDEEnv, strategy::MultiCenteredWithIndexObservation)
    minisections = strategy.minisections
    obs_length = 2 * minisections + 3
    low = [0.0f0 for _ in 1:obs_length]
    high = [1.0f0 for _ in 1:obs_length]
    return DRiL.Box(low, high)
end

function _multi_agent_action_space(::RDEEnv, ::VectorPressureAction)
    return DRiL.Box([-1.0f0], [1.0f0])
end

function _multi_agent_action_space(::RDEEnv, ::LinearVectorPressureAction)
    return DRiL.Box([-1.0f0], [1.0f0])
end

function _multi_agent_action_space(env::RDEEnv, ::DirectVectorPressureAction)
    return DRiL.Box([0.0f0], [Float32(env.u_pmax)])
end

struct MultiAgentRDEEnv{T, A, O, RW, G, V, OBS, M, RS, C} <: DRiL.AbstractParallelEnv
    core_env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}
    observation_space::DRiL.Box
    action_space::DRiL.Box
    n_envs::Int
    function MultiAgentRDEEnv(core_env::RDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}) where {T, A, O, RW, G, V, OBS, M, RS, C}
        obs_strategy = core_env.observation_strat
        @assert obs_strategy isa AbstractMultiAgentObservationStrategy
        @assert core_env.action_strat isa AbstractVectorActionStrategy
        action_strat = core_env.action_strat
        observation_space = _observation_space(core_env, obs_strategy)
        action_space = _multi_agent_action_space(core_env, action_strat)
        n_envs = obs_strategy.n_sections
        @assert n_envs == action_strat.n_sections
        return new{T, A, O, RW, G, V, OBS, M, RS, C}(core_env, observation_space, action_space, n_envs)
    end
end

RDE_Env.set_target_shock_count!(env::MultiAgentRDEEnv, target_shock_count::Int) = set_target_shock_count!(env.core_env, target_shock_count)
RDE_Env.get_target_shock_count(env::MultiAgentRDEEnv) = get_target_shock_count(env.core_env)

function DRiL.reset!(env::MultiAgentRDEEnv)
    return _reset!(env.core_env)
end

function DRiL.observe(env::MultiAgentRDEEnv)
    observation_matrix = _observe(env.core_env)
    return collect.(eachslice(observation_matrix, dims = ndims(observation_matrix)))
end

function DRiL.act!(env::MultiAgentRDEEnv{T, A, O, RW, G, V, OBS, M, RS, C}, actions::AbstractVector{<:AbstractVector}) where {T, A, O, RW, G, V, OBS, M, RS, C}
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
