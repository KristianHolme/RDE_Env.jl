# RDE_Env

[![Build Status](https://github.com/KristianHolme/RDE_Env.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KristianHolme/RDE_Env.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia module providing a reinforcement learning environment interface for the Rotating Detonation Engine (RDE) simulation.

## Overview

RDE_Env wraps the RDE simulation in a reinforcement learning environment with package extensions for both DRiL.jl and CommonRLInterface.jl. It provides:
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
    dt=0.5f0,                                    # Time step
    τ_smooth=0.01f0,                             # Control smoothing time constant
    observation_strat=FourierObservation(16),  # Use 16 Fourier modes as observation
    action_strat=ScalarPressureAction(),          # Single pressure control
    reward_strat=PeriodicityReward()              # Reward based on wave stability
)

# Run a random policy
policy = RandomRDEPolicy(env)
data = run_policy(policy, env)

# Visualize results
plot_policy_data(env, data)
```

## Components

### Action Types
- `ScalarPressureAction`: Single pressure control
- `ScalarAreaScalarPressureAction`: Independent area and pressure control
- `VectorPressureAction`: Multiple pressure controls along the domain (for multi-agent setups)
- `PIDAction`: PID controller-based action

### Observation Strategies

**Single-Agent:**
- `StateObservation`: Full state observation
- `FourierObservation`: Fourier modes of the state
- `SampledStateObservation`: Sampled points from the state
- `SectionedStateObservation`: State divided into sections with aggregated features
- `CompositeObservation`: FFT features + shock count + span metrics
- `MeanInjectionPressureObservation`: Mean injection pressure only

**Multi-Agent:**
- `MultiSectionObservation`: Each agent observes a section with look-ahead
- `MultiCenteredObservation`: Each agent observes the full domain centered on their section

### Reward Types

**Single-Agent:**
- `PeriodicityReward`: Rewards stable wave patterns
- `ShockSpanReward`: Rewards specific shock spacing
- `ShockPreservingReward`: Rewards maintaining shock count
- `PeriodMinimumReward`: Rewards based on period-averaged minimum pressure
- `CompositeReward`: Combination of multiple reward components
- `ExponentialAverageReward`: Exponentially weighted reward averaging
- `TransitionBasedReward`: Rewards based on state transitions

**Multi-Agent:**
- `MultiSectionReward`: Section-wise composite rewards
- `MultiSectionPeriodMinimumReward`: Period minimum reward per section

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

### DRiL.jl Integration

RDE_Env provides a package extension for seamless integration with [DRiL.jl](https://github.com/KristianHolme/DRiL.jl):

```julia
using RDE_Env
using DRiL  # This automatically loads the extension

# Create base environment
base_env = RDEEnv(
    observation_strat = FourierObservation(16),
    action_strat = ScalarPressureAction(),
    params = RDEParam(N = 512, tmax = 100.0f0)
)

# Wrap for DRiL interface
DRiLExt = Base.get_extension(RDE_Env, :RDE_EnvDRiLExt)
env = DRiLExt.DRiLRDEEnv(base_env)

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
    observation_strat = MultiCenteredObservation(n_sections = 4),
    action_strat = VectorPressureAction(n_sections = 4),
    reward_strat = MultiSectionPeriodMinimumReward(n_sections = 4),
    params = RDEParam(N = 512, tmax = 100.0f0)
)

# Wrap for DRiL multi-agent interface
DRiLExt = Base.get_extension(RDE_Env, :RDE_EnvDRiLExt)
env = DRiLExt.DRiLMultiAgentRDEEnv(base_env)

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
plot_policy_data(env, data)

# plot the whole simulation in a moving referance frame
plot_shifted_history(data, env.prob.x)

# Animated visualization
animate_policy_data(env, data)
```