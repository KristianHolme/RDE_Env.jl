using BenchmarkTools
using RDE_Env
using RDE_Env.Drill

include("bench_utils.jl")
using .BenchUtils

const SUITE = BenchmarkGroup()

env_api = BenchmarkGroup()
SUITE["env_api"] = env_api

env_api["act!"] = @benchmarkable begin
    Drill.act!(env, action)
end setup = begin
    env, rng = BenchUtils.setup_env()
    action = BenchUtils.setup_action(env, rng)
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES

env_api["observe"] = @benchmarkable begin
    Drill.observe(env)
end setup = begin
    env, rng = BenchUtils.setup_env()
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES

env_api["reset!"] = @benchmarkable begin
    Drill.reset!(env)
end setup = begin
    env, rng = BenchUtils.setup_env()
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES

utils = BenchmarkGroup()
SUITE["utils"] = utils

utils["get_avg_wave_speed"] = @benchmarkable begin
    get_avg_wave_speed(us, ts, dx)
end setup = begin
    data, us, ts, dx = BenchUtils.setup_policy_data()
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES

utils["speed_tracking"] = @benchmarkable begin
    RDE_Env.speed_tracking(data, dx)
end setup = begin
    data, us, ts, dx = BenchUtils.setup_policy_data()
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES

utils["get_plotting_speed_adjustments"] = @benchmarkable begin
    get_plotting_speed_adjustments(data, dx; max_speed = max_speed)
end setup = begin
    data, us, ts, dx = BenchUtils.setup_policy_data()
    max_speed = 4.6f0
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES

utils["adjust_for_jumps!"] = @benchmarkable begin
    local_speeds = copy(jump_speeds)
    RDE_Env.adjust_for_jumps!(local_speeds, max_speed; fallback_speed = fallback_speed)
end setup = begin
    jump_speeds = BenchUtils.setup_jump_speeds()
    max_speed = 4.6f0
    fallback_speed = 1.71f0
end evals = 1 samples = BenchUtils.DEFAULT_SAMPLES
