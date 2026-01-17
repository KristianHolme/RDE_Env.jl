# RDE_Env

[![Build Status](https://github.com/KristianHolme/RDE_Env.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KristianHolme/RDE_Env.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/JET.jl-enabled-blue)](https://github.com/aviatesk/JET.jl)

RDE_Env wraps the Rotating Detonation Engine (RDE) simulator from [RDE.jl](https://github.com/KristianHolme/RDE.jl) as a reinforcement learning environment built for [DRiL.jl](https://github.com/KristianHolme/DRiL.jl).

It provides:
- `RDEEnv` implementing the DRiL environment interface
- Swappable strategies (action/observation/reward/context) with optional caching
- `run_policy` for evaluating any `DRiL.AbstractPolicy`
- Makie-based plotting and animation utilities
- And more...

## Installation

```julia
] add https://github.com/KristianHolme/RDE_Env.jl
```
or
```julia
using Pkg
Pkg.add(url="https://github.com/KristianHolme/RDE_Env.jl")
```

## Quickstart (single-agent)

```julia
using DRiL
using RDE_Env

env = RDEEnv(;
    dt = 0.5f0,
    τ_smooth = 0.01f0,
    action_strat = DirectScalarPressureAction(),
    observation_strat = FullStateObservation(),
    reward_strat = USpanReward(),
    context_strat = NoContextStrategy(),
)

DRiL.reset!(env)
obs = DRiL.observe(env)
action = rand(DRiL.action_space(env))
reward = DRiL.act!(env, action)
```

### Multi-agent wrapper
To use a multi-agent interface, build a base `RDEEnv` with a multi-agent observation strategy and a vector action strategy, then wrap it in `MultiAgentRDEEnv`:

```julia
using RDE_Env

base_env = RDEEnv(;
    observation_strat = FullStateCenteredObservation(n_sections = 4),
    action_strat = DirectVectorPressureAction(n_sections = 4),
    reward_strat = ScalarToVectorReward(USpanReward(), 4),
)
ma_env = MultiAgentRDEEnv(base_env)
```

## Interfaces (strategies + caching)

Strategies are regular Julia types that specialize a few methods. Most hooks receive a `context::AbstractCache` argument, which lets action/observation/reward share state.

### Caching model
- `initialize_cache(strategy, N, T) -> cache::AbstractCache` (default returns `NoCache()`).
- `reset_cache!(cache)` (default is no-op).
- Caches live in `env.cache.*`:
  - `env.cache.action_cache`
  - `env.cache.observation_cache`
  - `env.cache.reward_cache`
  - `env.cache.context`

### Action strategies
Subtype `AbstractScalarActionStrategy` or `AbstractVectorActionStrategy`.

Implement:
- `RDE_Env.apply_action!(env, action, action_strat, action_strat_cache, context::AbstractCache)`
- `RDE_Env._action_space(env, action_strat)`

Optional:
- `initialize_cache(action_strat, N, T)`, `reset_cache!`

### Observation strategies
Subtype `AbstractObservationStrategy` (or `AbstractMultiAgentObservationStrategy`).

Implement:
- `RDE_Env.get_init_observation(strategy, N, T)`
- `RDE_Env.compute_observation!(obs, env, strategy, context::AbstractCache)`

If using custom cache, implement:
- `RDE_Env._observation_space(env, strategy)`

Optional:
- `initialize_cache(strategy, N, T)`, `reset_cache!`

### Reward strategies
Subtype `AbstractScalarRewardStrategy` or `AbstractVectorRewardStrategy`.

Implement:
- `RDE_Env.compute_reward(env, rew_strat, reward_cache, context::AbstractCache)`

Require:
- Subtype either `AbstractScalarRewardStrategy` or `AbstractVectorRewardStrategy`.

If using custom cache, implement:
- `initialize_cache(rew_strat, N, T)`, `reset_cache!`

Rewards may also set termination/truncation flags and store diagnostics in `env.info`.

### Context strategies (shared state)
Subtype `AbstractContextStrategy`.

If using custom cache, implement:
- `initialize_cache(context_strat, N, T)` (default `NoCache()`)
- `on_reset!(context_cache, context_strat, env)`
- `on_step!(context_cache, context_strat, env)` (default no-op)

## Custom implementations

### Custom context strategy

```julia
using RDE_Env

mutable struct TargetContext <: AbstractCache
    target::Int
end

struct FixedTargetContextStrategy <: AbstractContextStrategy
    target::Int
end

function RDE_Env.initialize_cache(cs::FixedTargetContextStrategy, ::Int, ::Type{T}) where {T}
    return TargetContext(cs.target)
end

function RDE_Env.on_reset!(cache::TargetContext, cs::FixedTargetContextStrategy, ::AbstractRDEEnv)
    cache.target = cs.target
    return nothing
end
```

### Custom observation strategy

```julia
using RDE
using RDE_Env

struct USpanAndTargetObservation <: AbstractObservationStrategy end

function RDE_Env.get_init_observation(::USpanAndTargetObservation, ::Int, ::Type{T}) where {T <: AbstractFloat}
    return Vector{T}(undef, 2)
end

function RDE_Env.compute_observation!(
        obs,
        env::RDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C},
        ::USpanAndTargetObservation,
        context::AbstractCache,
    ) where {T, A, O, RW, CS, V, OBS, M, RS, C}
    N = env.prob.params.N
    u = @view env.state[1:N]
    u_min, u_max = RDE.turbo_extrema(u)
    obs[1] = u_max - u_min
    obs[2] = context isa TargetContext ? T(context.target) : zero(T)
    return obs
end
```

### Custom action strategy

```julia
using RDE_Env
using DRiL: Box
struct TargetOffsetPressureAction <: AbstractScalarActionStrategy end

RDE_Env._action_space(::RDEEnv, ::TargetOffsetPressureAction) = Box([-1.0], [1.0])

function RDE_Env.apply_action!(
        env::RDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C},
        action::T,
        ::TargetOffsetPressureAction,
        ::AbstractCache,
        context::AbstractCache,
    ) where {T <: AbstractFloat, A <: TargetOffsetPressureAction, O, RW, CS, V, OBS, M, RS, C}
    offset = context isa TargetContext ? T(context.target) * T(0.01) : zero(T)
    u_p = clamp(action + offset, zero(T), env.u_pmax)
    method_cache = env.prob.method.cache
    copyto!(method_cache.u_p_previous, method_cache.u_p_current)
    method_cache.u_p_current .= u_p
    copyto!(method_cache.s_previous, method_cache.s_current)
    return nothing
end
```

### Custom reward strategy

```julia
using RDE
using RDE_Env

struct TargetScaledUSpanReward <: AbstractScalarRewardStrategy end

function RDE_Env.compute_reward(
        env::RDEEnv{T, A, O, RW, CS, V, OBS, M, RS, C},
        ::TargetScaledUSpanReward,
        ::NoCache,
        context::AbstractCache,
    ) where {T, A, O, RW, CS, V, OBS, M, RS, C}
    N = env.prob.params.N
    u = @view env.state[1:N]
    u_min, u_max = RDE.turbo_extrema(u)
    scale = context isa TargetContext ? max(one(T), T(context.target)) : one(T)
    return (u_max - u_min) / scale
end
```

### Custom policy (DRiL)
`run_policy` collects a trajectory from the environment using actions from the supplied policy at every step. It accepts any `policy::DRiL.AbstractPolicy` and calls it as `policy(obs; deterministic = true)`.

```julia
using DRiL

struct ConstantBoxPolicy{T} <: DRiL.AbstractPolicy
    action::T
end

function (π::ConstantBoxPolicy)(obs; deterministic::Bool = true)
    return [π.action]
end

policy = ConstantBoxPolicy([0.65f0])
env = RDEEnv()
data = run_policy(policy, env)

fig = plot_shifted_history(data, env.prob.x)
```

## Training with DRiL

```julia
using DRiL
using RDE_Env

function make_env()
    return RDEEnv(;
        dt = 1.0f0,
        action_strat = DirectScalarPressureAction(),
        observation_strat = FullStateObservation(),
        reward_strat = USpanReward(),
    )
end

env = BroadcastedParallelEnv([make_env() for _ in 1:16])
env = MonitorWrapperEnv(env)

alg = DRiL.PPO()
policy = ActorCriticPolicy(observation_space(env), action_space(env))
agent = ActorCriticAgent(policy; verbose = 2)

learn_stats, to = learn!(agent, env, alg, 100_000)
```

## Evaluation with `run_policy`

```julia
using DRiL
using RDE_Env

env = RDEEnv()
policy = DRiL.RandomPolicy(env)
data = run_policy(policy, env; saves_per_action = 10)
```

## Plotting (Makie)
Several plotting functions are available:

- `plot_policy_data(data, env; kwargs...)`: interactive figure for a recorded rollout (`PolicyRunData`) with a time scrubber and optional panels (rewards, controls, observations, etc.). This is the main entry point.
- `plot_shifted_history(data, env.prob.x; kwargs...)`: Static plot of the trajectory. Optionally includes a moving reference frame (useful for wave/shock visualization).
- `plot_policy(policy, env)`: convenience wrapper that runs `run_policy(policy, env)` and then calls `plot_policy_data`.
- `animate_policy(policy, env)`: convenience wrapper that runs a policy and animates it.
- `animate_policy_data(env, data)`: animate an already-recorded rollout.

For full details, use Julia help mode in the REPL:
`?plot_policy_data`, `?plot_shifted_history`, `?plot_policy`, `?animate_policy`, `?animate_policy_data`.


```julia
using RDE_Env

env = RDEEnv()
policy = DRiL.RandomPolicy(env)
data = run_policy(policy, env; saves_per_action = 10)

fig = plot_policy_data(data, env)

# Other helpers:
# - plot_shifted_history(data, env.prob.x)
# - plot_policy(policy, env)
# - animate_policy(policy, env)
# - animate_policy_data(env, data)
```

## Interactive control

```julia
using GLMakie
using RDE_Env

env = RDEEnv()
interactive_control(env)
```
