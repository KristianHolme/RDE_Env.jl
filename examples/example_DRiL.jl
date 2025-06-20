using DRiL
using RDE_Env
DRiLRDE = Base.get_extension(RDE_Env, :RDE_EnvDRiLExt)

env = BroadcastedParallelEnv([DRiLRDE.DRiLRDEEnv(RDEEnv(dt=1f0)) for _ in 1:16])

alg = DRiL.PPO()
env = MonitorWrapperEnv(env)
env = NormalizeWrapperEnv(env, gamma=alg.gamma)

policy = ActorCriticPolicy(observation_space(env), action_space(env))
agent = ActorCriticAgent(policy; verbose=2, n_steps=128, batch_size=128, learning_rate=3f-4, epochs=10)
## train agent
learn_stats = learn!(agent, env, alg; max_steps=100_000)
## wrap agent in DRiLAgentPolicy
policy = DRiLAgentPolicy(agent, env)
## run policy
data = run_policy(policy, env)
##
raw_envs = unwrap_all(env) #this is a parallel env
x = raw_envs.envs[1].prob.x
fig = plot_shifted_history(data, x)





