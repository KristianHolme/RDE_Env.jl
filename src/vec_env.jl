using CommonRLInterface
using RDE
using Polyester
"""
    ThreadingMode

Enum for different threading modes available in the vectorized environment.
"""
@enum ThreadingMode begin
    THREADS  # Use Base.Threads.@threads
    POLYESTER  # Use Polyester.@batch
end
"""
    RDEVecEnv{E<:AbstractEnv}

A vectorized environment that runs multiple RDE environments in parallel.
Implements the CommonRLInterface and provides a Stable Baselines compatible step! function.

The environment supports two threading modes:
- THREADS: Uses Base.Threads.@threads for parallelization
- POLYESTER: Uses Polyester.@batch for parallelization
"""
mutable struct RDEVecEnv{E<:AbstractEnv} <: AbstractEnv
    envs::Vector{E}
    n_envs::Int
    observations::Matrix{Float32}  # Pre-allocated for efficiency
    rewards::Vector{Float32}
    dones::Vector{Bool}
    infos::Vector{Dict{String,Any}}
    reset_infos::Vector{Dict{String,Any}}
    threading_mode::ThreadingMode
end

"""
    RDEVecEnv(envs::Vector{<:AbstractEnv}; threading_mode=THREADS)

Create a vectorized environment from a vector of environments.
"""
function RDEVecEnv(envs::Vector{E}; threading_mode::ThreadingMode=THREADS) where {E<:AbstractEnv}
    n_envs = length(envs)
    obs_dim = length(CommonRLInterface.observe(envs[1]))
    observations = Matrix{Float32}(undef, obs_dim, n_envs)
    rewards = zeros(Float32, n_envs)
    dones = fill(false, n_envs)
    infos = [Dict{String,Any}() for _ in 1:n_envs]
    reset_infos = [Dict{String,Any}() for _ in 1:n_envs]
    
    RDEVecEnv{E}(envs, n_envs, observations, rewards, dones, infos, reset_infos, threading_mode)
end

function Base.show(io::IO, env::RDEVecEnv)
    println(io, "RDEVecEnv with $(env.n_envs) environments using $(env.threading_mode) threading:")
    for i in 1:env.n_envs
        println(io, "  Environment $i:")
        show(io, env.envs[i])
    end
end

# Helper function to reset a single environment
function reset_single_env!(env::RDEVecEnv, i::Int)
    CommonRLInterface.reset!(env.envs[i])
    env.observations[:, i] .= CommonRLInterface.observe(env.envs[i])
    env.reset_infos[i] = Dict{String,Any}()
end

# Helper function to act on a single environment
function act_single_env!(env::RDEVecEnv, i::Int, actions::AbstractArray)
    # Step environment
    env.rewards[i] = CommonRLInterface.act!(env.envs[i], @view actions[:, i])
    @logmsg LogLevel(-10000) "VecEnv act! done with env $i, starting termination check"
    
    # Check termination
    if CommonRLInterface.terminated(env.envs[i])
        env.dones[i] = true
        env.infos[i]["terminal_observation"] = CommonRLInterface.observe(env.envs[i])
        if env.envs[i].truncated  # TODO: Add truncated to CommonRLInterface?
            env.infos[i]["TimeLimit.truncated"] = true
        end
        CommonRLInterface.reset!(env.envs[i])
    else
        env.dones[i] = false
        empty!(env.infos[i])
    end
    
    # Update observation
    @logmsg LogLevel(-10000) "VecEnv act! done with env $i, starting observation update"
    env.observations[:, i] .= CommonRLInterface.observe(env.envs[i])
    @logmsg LogLevel(-10000) "VecEnv act! done with env $i, observation update done"
end

# CommonRLInterface implementations
function CommonRLInterface.reset!(env::RDEVecEnv)
    # Choose threading macro based on mode
    if env.threading_mode == THREADS
        Threads.@threads for i in 1:env.n_envs
            reset_single_env!(env, i)
        end
    else  # POLYESTER
        @batch for i in 1:env.n_envs
            reset_single_env!(env, i)
        end
    end
    nothing
end

function CommonRLInterface.observe(env::RDEVecEnv)
    copy(env.observations)
end

function CommonRLInterface.act!(env::RDEVecEnv, actions::AbstractArray)
    @logmsg LogLevel(-10000) "VecEnv act! starting threaded loop, actions size: $(size(actions))"
    @assert size(actions, 2) == env.n_envs && size(actions, 1) == action_dim(env.envs[1].action_type) "Action size mismatch"
    
    # Choose threading macro based on mode
    if env.threading_mode == THREADS
        Threads.@threads for i in 1:env.n_envs
            act_single_env!(env, i, actions)
        end
    else  # POLYESTER
        @batch for i in 1:env.n_envs
            act_single_env!(env, i, actions)
        end
    end
    
    copy(env.rewards)
end

function CommonRLInterface.terminated(env::RDEVecEnv) #this is SB done, not terminated
    copy(env.dones)
end

"""
    seed!(env::AbstractEnv, seed::Int)

Set the random seed for the environment.
"""
function seed!(env::AbstractEnv, seed::Int)
    Random.seed!(seed)
end

"""
    step!(env::RDEVecEnv, actions::AbstractMatrix)

Step all environments in parallel with the given actions.
Returns (observations, rewards, dones, infos) matching the Stable Baselines API.
"""
function step!(env::RDEVecEnv, actions::AbstractArray)
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