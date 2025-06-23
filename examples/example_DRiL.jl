using DRiL
using RDE_Env
using Zygote
DRiLRDE = Base.get_extension(RDE_Env, :RDE_EnvDRiLExt)
##
function make_RDE_Env()
    target_shock_count = 3
    env = RDEEnv(dt=1f0,
        reset_strategy=RandomShock(),
        action_type=ScalarPressureAction(),
        observation_strategy=SectionedStateObservation(target_shock_count=target_shock_count),
        reward_type=PeriodMinimumReward(target_shock_count=target_shock_count))
    _reset!(env)
    return env
end
##
env = BroadcastedParallelEnv([DRiLRDE.DRiLRDEEnv(make_RDE_Env()) for _ in 1:16])

alg = DRiL.PPO()
env = MonitorWrapperEnv(env)
env = NormalizeWrapperEnv(env, gamma=alg.gamma)

policy = ActorCriticPolicy(observation_space(env), action_space(env))
agent = ActorCriticAgent(policy; verbose=2, n_steps=4, batch_size=32, learning_rate=3f-4, epochs=10)
## train agent
learn_stats = learn!(agent, env, alg; max_steps=100_000)
## wrap agent in DRiLAgentPolicy
policy = DRiLRDE.DRiLAgentPolicy(agent, env)
## run policy
single_env = unwrap_all(env).envs[1]
data = run_policy(policy, single_env)
##
x = single_env.prob.x
fig = plot_shifted_history(data, x)





