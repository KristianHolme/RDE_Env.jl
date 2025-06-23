# RDE_Env

A Julia module providing a reinforcement learning environment interface for the Rotating Detonation Engine (RDE) simulation.

## Overview

RDE_Env wraps the RDE simulation in a reinforcement learning environment with extensions for the DRiL.jl interface and the CommonRLInterface.jl interface. It provides:
- Various action spaces for controlling RDE parameters
- Multiple observation strategies
- Customizable reward functions
- Vectorized environments for parallel training
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
    observation_strategy=FourierObservation(16),  # Use 16 Fourier modes as observation
    action_type=ScalarPressureAction(),          # Single pressure control
    reward_type=PeriodicityReward()              # Reward based on wave stability
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
- `VectorPressureAction`: Multiple pressure controls along the domain

### Observation Strategies
- `StateObservation`: Full state observation
- `FourierObservation`: Fourier modes of the state
- `SampledStateObservation`: Sampled points from the state
- `SectionedStateObservation`: State divided into sections, for single-agent environments
- `MultiSectionObservation`: State divided into sections, for Multi-agent environments
- `CompositeObservation`: Combination of state, number of shocks, target number of shocks, and span of pressure

### Reward Types
- `PeriodicityReward`: Rewards stable wave patterns
- `ShockSpanReward`: Rewards specific shock spacing
- `ShockPreservingReward`: Rewards maintaining shock count
- `MultiSectionReward`: Similar to CompositeReward, but for Multi-agent environments
- `CompositeReward`: Combination of multiple rewards, rewarding stability, periodicity, and distance to target number of shocks, and span of pressure

### Policies
- `StepwiseRDEPolicy`: Predefined control sequence
- `RandomRDEPolicy`: Random actions
- `ConstantRDEPolicy`: Constant controls
- `SinusoidalRDEPolicy`: Sinusoidal control patterns
- `DelayedPolicy`: Wrapper to delay policy activation

### Vectorized Environments
- `RDEVecEnv`: Parallel environments for faster training
- `MultiAgentRDEVecEnv`: Multi-agent parallel environments

## Advanced Features

### Control Smoothing
The environment supports smooth control transitions through the `τ_smooth` parameter:
```julia
env = RDEEnv(dt=0.5f0, τ_smooth=0.01f0)  # τ_smooth should be < dt
```

### Action Momentum
Control changes can be smoothed using momentum:
```julia
env = RDEEnv(dt=0.5f0, momentum=0.8f0)  # 0.8 momentum factor
```

### Interactive Control
For debugging and exploration:
```julia
interactive_control(env)  # Opens interactive control GUI (requires GLMakie or WGLMakie)
```

## Visualization

The module provides several visualization tools:
```julia
# Basic trajectory plot
plot_policy_data(env, data)

# plot the whole simulation in a moving referance frame
plot_shifted_history(data, env.prob.x)

# Animated visualization
animate_policy_data(env, data)
```