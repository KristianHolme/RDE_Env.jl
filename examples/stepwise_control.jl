using RDE
using RDE_Env
using GLMakie

params = RDEParam(tmax = 500.0f0, N = 2048)
env = RDEEnv(
    params, method = FiniteVolumeMethod(),
    action_type = ScalarAreaScalarPressureAction()
);
π = StepwiseRDEPolicy(
    env, [20.0f0, 100.0f0, 200.0f0, 350.0f0],
    [[3.5f0, 0.64f0], [3.5f0, 0.86f0], [3.5f0, 0.64f0], [3.5f0, 0.94f0]]
);
data = run_policy(π, env)
fig = plot_policy_data(env, data)
##
animate_policy_data(data, env; fname = "stepwise_control", fps = 60)
