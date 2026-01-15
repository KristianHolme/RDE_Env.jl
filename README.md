# RDE_Env

[![Build Status](https://github.com/KristianHolme/RDE_Env.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KristianHolme/RDE_Env.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia module providing a reinforcement learning environment interface for the Rotating Detonation Engine (RDE) simulation.

## Overview

RDE_Env wraps the RDE simulation in a reinforcement learning environment built for DRiL.jl. It provides:
- Various action spaces for controlling RDE parameters
- Multiple observation strategies
- Customizable reward functions
- Single-agent and multi-agent environment support
- Policy implementations and evaluation tools
- Visualization utilities

## Installation

```julia
] add https://github.com/KristianHolme/RDE_Env.jl
```

## Basic Usage

```julia
using RDE_Env

# Create a basic environment
env = RDEEnv(
    dt=0.5f0,                                  # Time step
    τ_smooth=0.01f0,                           # Control smoothing time constant
    observation_strat=FullStateObservation(), # Raw u, λ state
    action_strat=DirectScalarPressureAction(), # Direct pressure control
    reward_strat=USpanReward()                 # Span of u
)

# Run a random policy
policy = RandomRDEPolicy(env)
data = run_policy(policy, env)

# Visualize results
plot_policy_data(data, env)
```

## Components

### Action Types
- `DirectScalarPressureAction`: Direct scalar pressure control
- `DirectVectorPressureAction`: Direct per-section pressure control

### Observation Strategies

**Single-Agent:**
- `FullStateObservation`: Raw `[u; λ]` state

**Multi-Agent:**
- `FullStateCenteredObservation`: Raw full state centered per section

### Reward Types

**Single-Agent:**
- `USpanReward`: Rewards span of `u`

**Multi-Agent:**
- `ScalarToVectorReward`: Wrap a scalar reward into per-agent values

### Policies
- `StepwiseRDEPolicy`: Predefined control sequence
- `RandomRDEPolicy`: Random actions
- `ConstantRDEPolicy`: Constant controls
- `SinusoidalRDEPolicy`: Sinusoidal control patterns
- `LinearPolicy`: Linear interpolation between control points
- `SawtoothPolicy`: Sawtooth pressure control pattern
- `PIDControllerPolicy`: PID controller for shock count stabilization
- `DelayedPolicy`: Wrapper to delay policy activation

## Advanced Features
# Extra strategy variants

Additional action, observation, and reward strategy variants are maintained in the project repository and are available via `DRL_RDE.RDE_Env_Strategies`.

### DRiL.jl Integration

`RDEEnv` directly implements the [DRiL.jl](https://github.com/KristianHolme/DRiL.jl) environment interface:

```julia
using RDE_Env
using DRiL

# Create base environment
env = RDEEnv(
    observation_strat = FourierObservation(16),
    action_strat = ScalarPressureAction(),
    params = RDEParam(N = 512, tmax = 100.0f0)
)

# Create parallel environment for training
parallel_env = BroadcastedParallelEnv([env for _ in 1:4])
parallel_env = MonitorWrapperEnv(parallel_env)
parallel_env = NormalizeWrapperEnv(parallel_env)

# Create policy
policy = ActorCriticPolicy(observation_space(env), action_space(env))

# Create PPO algorithm and agent
alg = PPO(n_steps=2048, batch_size=64, epochs=10, learning_rate=3f-4)
agent = ActorCriticAgent(policy, alg; verbose=2)

# Train the agent
max_steps = 100_000
learn_stats, to = learn!(agent, parallel_env, alg, max_steps)
```

### Multi-Agent Environments

Create multi-agent environments with section-wise control:

```julia
using RDE_Env
using DRiL

# Create multi-agent base environment
base_env = RDEEnv(
    observation_strat = FullStateCenteredObservation(n_sections = 4),
    action_strat = DirectVectorPressureAction(n_sections = 4),
    reward_strat = ScalarToVectorReward(USpanReward(), 4),
    params = RDEParam(N = 512, tmax = 100.0f0)
)

# Wrap for DRiL multi-agent interface
env = MultiAgentRDEEnv(base_env)

# Create parallel multi-agent environment (multiple instances of multi-agent env)
parallel_env = MultiAgentParallelEnv([env for _ in 1:4])
parallel_env = MonitorWrapperEnv(parallel_env)
parallel_env = NormalizeWrapperEnv(parallel_env)

# Create policy and agent
policy = ActorCriticPolicy(observation_space(env), action_space(env))
alg = PPO(n_steps=2048, batch_size=64, epochs=10)
agent = ActorCriticAgent(policy, alg; verbose=2)

# Train
max_steps = 200_000
learn_stats, to = learn!(agent, parallel_env, alg, max_steps)
```


### Interactive Control
For debugging and exploration:
```julia
interactive_control(env)  # Opens interactive control GUI (requires GLMakie or WGLMakie)
```

## Visualization

The module provides several visualization tools:
```julia
# Basic trajectory plot (requires GLMakie or WGLMakie)
plot_policy_data(data, env)

# plot the whole simulation in a moving referance frame
plot_shifted_history(data, env.prob.x)

# Animated visualization
animate_policy_data(env, data)
```