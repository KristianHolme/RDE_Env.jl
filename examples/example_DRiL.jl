using DRiL
using RDE_Env
using Zygote
DRiLRDE = Base.get_extension(RDE_Env, :RDE_EnvDRiLExt)
##
function make_RDE_Env(tmax::Float32 = 50.0f0, target_shock_count::Int = 3)
    params = RDEParam(; tmax = tmax)
    env = RDEEnv(
        params; dt = 1.0f0,
        reset_strategy = RandomShock(),
        action_type = ScalarPressureAction(),
        observation_strategy = SectionedStateObservation(target_shock_count = target_shock_count),
        reward_type = MultiplicativeReward(
            PeriodMinimumReward(target_shock_count = target_shock_count, lowest_action_magnitude_reward = 0.5f0),
            TimeDiffNormReward()
        )
    )
    _reset!(env)
    return env
end
##
env = BroadcastedParallelEnv([DRiLRDE.DRiLRDEEnv(make_RDE_Env(60.0f0, 3)) for _ in 1:16])

alg = DRiL.PPO()
env = MonitorWrapperEnv(env)
# env = NormalizeWrapperEnv(env, gamma=alg.gamma)

policy = ActorCriticPolicy(observation_space(env), action_space(env))
agent = ActorCriticAgent(policy; verbose = 2, n_steps = 4, batch_size = 32, learning_rate = 3.0f-4, epochs = 10)
## train agent
learn_stats = learn!(agent, env, alg, 100_000)
## wrap agent in DRiLAgentPolicy
policy = DRiLRDE.DRiLAgentPolicy(agent, env isa NormalizeWrapperEnv ? env : nothing)
## run policy
single_env = make_RDE_Env(200.0f0)
data = run_policy(policy, single_env)
##
x = single_env.prob.x
fig = plot_shifted_history(data, x)
