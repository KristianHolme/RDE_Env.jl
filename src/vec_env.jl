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
Provides a Stable Baselines compatible step! function.

The environment supports two threading modes:
- THREADS: Uses Base.Threads.@threads for parallelization
- POLYESTER: Uses Polyester.@batch for parallelization
"""
mutable struct RDEVecEnv{T<:AbstractFloat} <: AbstractRDEEnv
    envs::Vector{RDEEnv{T, A, O, R}} where {A<:AbstractActionType, O<:AbstractObservationStrategy, R<:AbstractRDEReward}
    n_envs::Int64
    observations::Matrix{T}  # Pre-allocated for efficiency
    rewards::Vector{T}
    dones::Vector{Bool}
    infos::Vector{Dict{String,Any}}
    reset_infos::Vector{Dict{String,Any}}
    threading_mode::ThreadingMode
end

"""
    RDEVecEnv(envs::Vector{<:AbstractEnv}; threading_mode=THREADS)

Create a vectorized environment from a vector of environments.
"""
function RDEVecEnv(envs::Vector{RDEEnv{T, A, O, R}}; threading_mode::ThreadingMode=THREADS) where {T<:AbstractFloat, A<:AbstractActionType, O<:AbstractObservationStrategy, R<:AbstractRDEReward}
    n_envs = length(envs)
    obs_dim = length(_observe(envs[1]))
    observations = Matrix{T}(undef, obs_dim, n_envs)
    rewards = zeros(T, n_envs)
    dones = fill(false, n_envs)
    infos = [Dict{String,Any}() for _ in 1:n_envs]
    reset_infos = [Dict{String,Any}() for _ in 1:n_envs]
    
    RDEVecEnv{T}(envs, n_envs, observations, rewards, dones, infos, reset_infos, threading_mode)
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
    _reset!(env.envs[i])
    env.observations[:, i] .= _observe(env.envs[i])
    env.reset_infos[i] = Dict{String,Any}()
end

# Helper function to act on a single environment
function act_single_env!(env::RDEVecEnv, i::Int, actions::AbstractArray)
    # Step environment
    env.rewards[i] = _act!(env.envs[i], @view actions[:, i])
    @logmsg LogLevel(-10000) "VecEnv act! done with env $i, starting termination check"
    
    # Check termination
    if env.envs[i].terminated
        env.dones[i] = true
        env.infos[i]["terminal_observation"] = _observe(env.envs[i])
        if env.envs[i].truncated 
            env.infos[i]["TimeLimit.truncated"] = true
        end
       _reset!(env.envs[i])
    else
        env.dones[i] = false
        empty!(env.infos[i])
    end
    
    # Update observation
    @logmsg LogLevel(-10000) "VecEnv act! done with env $i, starting observation update"
    env.observations[:, i] .= _observe(env.envs[i])
    @logmsg LogLevel(-10000) "VecEnv act! done with env $i, observation update done"
end

function _reset!(env::RDEVecEnv)
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

_observe(env::RDEVecEnv) = copy(env.observations)

function _act!(env::RDEVecEnv, actions::AbstractArray)
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
    _act!(env, actions)
    @logmsg LogLevel(-10000) "VecEnv act! done, returning stuff"
    return (
        _observe(env),
        copy(env.rewards),
        copy(env.dones),
        copy(env.infos)
    )
end 