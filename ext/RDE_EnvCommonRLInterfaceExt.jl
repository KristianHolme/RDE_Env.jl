module RDE_EnvCommonRLInterfaceExt
#deprecated, dont use this anymore
using RDE_Env
using CommonRLInterface

struct CommonRDEEnv <: CommonRLInterface.AbstractEnv
    core_env::RDEEnv
end

CommonRLInterface.act!(env::CommonRDEEnv, action; kwargs...) = _act!(env.core_env, action; kwargs...)
CommonRLInterface.reset!(env::CommonRDEEnv) = _reset!(env.core_env)

# CommonRLInterface implementations
CommonRLInterface.state(env::CommonRDEEnv) = vcat(env.core_env.state, env.core_env.t)
CommonRLInterface.terminated(env::CommonRDEEnv) = env.core_env.done
function CommonRLInterface.observe(env::CommonRDEEnv)
    return compute_observation(env, env.core_env.observation_strategy)
end

function CommonRLInterface.actions(env::CommonRDEEnv)
    n = action_dim(env.core_env.action_type)
    return [(-1 .. 1) for _ in 1:n]
end

function CommonRLInterface.clone(env::CommonRDEEnv)
    env2 = deepcopy(env)
    @logmsg LogLevel(-10000) "env is copied!"
    return env2
end

function CommonRLInterface.setstate!(env::CommonRDEEnv, s)
    env.core_env.state = s[1:(end - 1)]
    return env.core_env.t = s[end]
end
end
