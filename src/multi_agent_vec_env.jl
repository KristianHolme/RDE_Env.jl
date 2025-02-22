"""
    MultiAgentRDEVecEnv{T<:AbstractFloat}

A vectorized environment that runs multiple RDE environments in parallel. 
Each environment has multiple agents, and the observations are concatenated for each agent.
Implements the CommonRLInterface and provides a Stable Baselines compatible step! function.

The environment supports two threading modes:
- THREADS: Uses Base.Threads.@threads for parallelization
- POLYESTER: Uses Polyester.@batch for parallelization
"""
mutable struct MultiAgentRDEVecEnv{T<:AbstractFloat} <: AbstractRDEEnv{T}
    envs::Vector{<:AbstractRDEEnv{T}}
    n_envs::Int
    n_agents_per_env::Int
    observations::Matrix{T}
    rewards::Vector{T}
    dones::Vector{Bool}
    infos::Vector{Dict{String,Any}}
    reset_infos::Vector{Dict{String,Any}}
    threading_mode::ThreadingMode
end

"""
    MultiAgentRDEVecEnv(envs::Vector{<:AbstractRDEEnv}; threading_mode=THREADS)

Create a vectorized environment from a vector of environments.
"""
function MultiAgentRDEVecEnv(envs::Vector{<:AbstractRDEEnv{T}}; threading_mode::ThreadingMode=THREADS) where {T<:AbstractFloat}
    n_envs = length(envs)
    obs = CommonRLInterface.observe(envs[1])
    obs_dim = size(obs, 1)
    n_agents_per_env = size(obs, 2)
    observations = Matrix{T}(undef, obs_dim, n_envs*n_agents_per_env)
    rewards = zeros(T, n_envs*n_agents_per_env)
    dones = fill(false, n_envs*n_agents_per_env)
    infos = [Dict{String,Any}() for _ in 1:n_envs*n_agents_per_env]
    reset_infos = [Dict{String,Any}() for _ in 1:n_envs*n_agents_per_env]
    
    MultiAgentRDEVecEnv{T}(envs, n_envs, n_agents_per_env, observations, rewards, dones, infos, reset_infos, threading_mode)
end

function Base.show(io::IO, env::MultiAgentRDEVecEnv)
    println(io, "MultiAgentRDEVecEnv with $(env.n_envs) environments using $(env.threading_mode) threading:")
    for i in 1:env.n_envs
        println(io, "  Environment $i:")
        show(io, env.envs[i])
    end
end

function env_indices(i::Int, n_agents_per_env::Int)
    return (1+(i-1)*n_agents_per_env):(i*n_agents_per_env)
end

# CommonRLInterface implementations
function CommonRLInterface.reset!(env::MultiAgentRDEVecEnv)
    num_agents = env.n_agents_per_env
    num_envs = env.n_envs
    
    # Choose threading macro based on mode
    if env.threading_mode == THREADS
        Threads.@threads for i in 1:num_envs
            reset_single_env!(env, i, num_agents)
        end
    else  # POLYESTER
        @batch for i in 1:num_envs
            reset_single_env!(env, i, num_agents)
        end
    end
    nothing
end

# Helper function to reset a single environment
function reset_single_env!(env::MultiAgentRDEVecEnv, i::Int, num_agents::Int)
    CommonRLInterface.reset!(env.envs[i])
    env.observations[:, env_indices(i, num_agents)] .= CommonRLInterface.observe(env.envs[i])
    env.reset_infos[i] = Dict{String,Any}()
end

function CommonRLInterface.observe(env::MultiAgentRDEVecEnv)
    copy(env.observations)
end

function CommonRLInterface.act!(env::MultiAgentRDEVecEnv, actions::AbstractArray)
    num_agents = env.n_agents_per_env
    num_envs = env.n_envs
    if length(actions) == num_envs*num_agents
        actions = reshape(actions, num_agents, num_envs)
    end
    @logmsg LogLevel(-10000) "VecEnv act! starting threaded loop, actions size: $(size(actions))"
    
    # Choose threading macro based on mode
    if env.threading_mode == THREADS
        Threads.@threads for i in 1:num_envs
            act_single_env!(env, i, num_agents, actions)
        end
    else  # POLYESTER
        @batch for i in 1:num_envs
            act_single_env!(env, i, num_agents, actions)
        end
    end
    
    copy(env.rewards)
end

# Helper function to act on a single environment
function act_single_env!(env::MultiAgentRDEVecEnv, i::Int, num_agents::Int, actions::AbstractArray)
    # Step environment
    env_inds = env_indices(i, num_agents)
    @logmsg LogLevel(-10000) "action size: $(size(actions))"
    env.rewards[env_inds] = CommonRLInterface.act!(env.envs[i], @view actions[:, i])
    @logmsg LogLevel(-10000) "VecEnv act! done with env $i, starting termination check"
    
    # Check termination
    if CommonRLInterface.terminated(env.envs[i])
        env.dones[env_inds] .= true
        for agent_i in 1:num_agents
            env.infos[env_inds[agent_i]]["terminal_observation"] = CommonRLInterface.observe(env.envs[i])[:, agent_i]
            if env.envs[i].truncated  # TODO: Add truncated to CommonRLInterface?
                env.infos[env_inds[agent_i]]["TimeLimit.truncated"] = true
            end
        end
        CommonRLInterface.reset!(env.envs[i])
    else
        env.dones[env_inds] .= false
        empty!.(env.infos[env_inds])
    end
    
    # Update observation
    @logmsg LogLevel(-10000) "VecEnv act! done with env $i, starting observation update"
    env.observations[:, env_indices(i, num_agents)] .= CommonRLInterface.observe(env.envs[i])
    @logmsg LogLevel(-10000) "VecEnv act! done with env $i, observation update done"
end

function CommonRLInterface.terminated(env::MultiAgentRDEVecEnv) #this is SB done, not terminated
    copy(env.dones)
end

"""
    step!(env::MultiAgentVecEnv, actions::AbstractMatrix)

Step all environments in parallel with the given actions.
Returns (observations, rewards, dones, infos) matching the Stable Baselines API.
"""
function step!(env::MultiAgentRDEVecEnv, actions::AbstractArray)
    @logmsg LogLevel(-10000) "VecEnv act!, actions size: $(size(actions))"
    CommonRLInterface.act!(env, actions)
    @logmsg LogLevel(-10000) "VecEnv act! done, returning stuff"
    return (
        CommonRLInterface.observe(env),
        copy(env.rewards),
        CommonRLInterface.terminated(env),
        copy(env.infos)
    )
end 