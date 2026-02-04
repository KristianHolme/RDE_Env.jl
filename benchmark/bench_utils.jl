module BenchUtils

using RDE_Env.DRiL
using Random
using RDE_Env.RDE: RDEParam, split_sol
using RDE_Env
using RDE_Env: DirectScalarPressureAction, FullStateObservation, RDEEnv, run_policy

const DEFAULT_SEED = 42
const DEFAULT_SAMPLES = 20
const DEFAULT_N = 128
const DEFAULT_TMAX = 0.3f0
const DEFAULT_DT = 0.1f0
const DEFAULT_TAU_SMOOTH = 0.05f0
const DEFAULT_SAVES_PER_ACTION = 1

function setup_env(;
        seed::Int = DEFAULT_SEED,
        N::Int = DEFAULT_N,
        tmax::Float32 = DEFAULT_TMAX,
        dt::Float32 = DEFAULT_DT,
        τ_smooth::Float32 = DEFAULT_TAU_SMOOTH,
    )
    rng = Random.Xoshiro(seed)
    params = RDEParam(; N = N, tmax = tmax)
    env = RDEEnv(;
        dt = dt,
        params = params,
        τ_smooth = τ_smooth,
        observation_strat = FullStateObservation(),
        action_strat = DirectScalarPressureAction(),
        verbose = false,
    )
    DRiL.reset!(env)
    return env, rng
end

function setup_action(env::RDEEnv, rng::AbstractRNG)
    action_space = DRiL.action_space(env)
    action = rand(rng, action_space)
    return action
end

function setup_policy_data(;
        seed::Int = DEFAULT_SEED,
        N::Int = DEFAULT_N,
        tmax::Float32 = DEFAULT_TMAX,
        dt::Float32 = DEFAULT_DT,
        τ_smooth::Float32 = DEFAULT_TAU_SMOOTH,
        saves_per_action::Int = DEFAULT_SAVES_PER_ACTION,
    )
    env, _ = setup_env(; seed = seed, N = N, tmax = tmax, dt = dt, τ_smooth = τ_smooth)
    policy = DRiL.RandomPolicy(env)
    data = run_policy(policy, env; saves_per_action = saves_per_action)
    us, _ = split_sol(data.states)
    ts = data.state_ts
    dx = env.prob.method.cache.dx
    return data, us, ts, dx
end

function setup_jump_speeds(; seed::Int = DEFAULT_SEED, n::Int = 64)
    rng = Random.Xoshiro(seed)
    speeds = abs.(randn(rng, Float32, n)) .+ 1.0f0
    if n >= 3
        speeds[cld(n, 2)] = 10.0f0
    end
    return speeds
end

end
