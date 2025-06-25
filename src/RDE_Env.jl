module RDE_Env
using RDE
using CircularArrays
using DomainSets
using FFTW
using LinearAlgebra
using Statistics
using Random
using Logging
using SciMLBase
using OrdinaryDiffEq
using Makie
using Observables
using Polyester
using POMDPTools
using PrecompileTools
using ProgressMeter

include("types.jl")
export AbstractRDEEnv, AbstractActionType, AbstractObservationStrategy, AbstractMultiAgentObservationStrategy,
    AbstractRDEReward
export action_dim, get_standard_normalized_actions, set_N!
export compute_observation, get_init_observation
export set_reward!
export RDEEnv, RDEEnvCache
include("utils.jl")
export sigmoid_to_linear, reward_sigmoid, sigmoid, linear_to_sigmoid

# Actions
include("actions.jl")
export ScalarPressureAction, ScalarAreaScalarPressureAction, VectorPressureAction

# Observation strategies
include("observations.jl")
export
    FourierObservation, StateObservation,
    SampledStateObservation, MultiSectionObservation, SectionedStateObservation,
    MultiCenteredObservation, MeanInjectionPressureObservation
export CompositeObservation, compute_sectioned_observation

# Rewards
include("rewards.jl")
export ShockSpanReward, ShockPreservingReward, ShockPreservingSymmetryReward
export CompositeReward, ConstantTargetReward, MultiSectionReward, PeriodicityReward
export TimeAggCompositeReward, TimeMin, TimeAvg, TimeMax, TimeSum, TimeProd, TimeAggMultiSectionReward
export TimeDiffNormReward, MultiplicativeReward, PeriodMinimumReward, PeriodMinimumVariationReward
export set_reward!, set_termination_reward!, compute_reward

# Environment
include("RLenv.jl")
export _reset!, _act!, _observe, state, terminated

include("policies.jl")
export AbstractRDEPolicy, StepwiseRDEPolicy, RandomRDEPolicy, ConstantRDEPolicy, SinusoidalRDEPolicy,
    DelayedPolicy, LinearPolicy, get_env, LinearCheckpoints, SawtoothPolicy, PIDControllerPolicy
export reset_pid_cache!, _predict_action
# Vectorized environments
include("vec_env.jl")
include("multi_agent_vec_env.jl")
export RDEVecEnv, MultiAgentRDEVecEnv
export step!, seed!
export ThreadingMode, POLYESTER, THREADS

# Policies
export PolicyRunData, run_policy
export ConstantRDEPolicy, SinusoidalRDEPolicy, StepwiseRDEPolicy, RandomRDEPolicy
export ScaledPolicy

# Plotting
include("plotting.jl")
export plot_policy_data, plot_shifted_history, plot_policy, animate_policy, animate_policy_data

include("interactive_control.jl")
export interactive_control

include("displaying.jl")

@compile_workload begin
    try
        # Create and run a small environment with random policy
        env = RDEEnv(;
            dt=0.01f0,
            params=RDEParam(; N=512, tmax=0.05f0),
            τ_smooth=0.001f0,
            momentum=0.8f0,
            observation_strategy=FourierObservation(8),
            action_type=ScalarPressureAction()
        )
        policy = RandomRDEPolicy(env)
        data = run_policy(policy, env, saves_per_action=2)

        # Test vectorized environment
        envs = [RDEEnv(;
            dt=0.01f0,
            τ_smooth=0.001f0,
            params=RDEParam(; N=512, tmax=0.05f0),
            observation_strategy=FourierObservation(8)
        ) for _ in 1:2]
        vec_env = RDEVecEnv(envs)
        _reset!(vec_env)
        actions = rand(Float32, 1, 2) .- 0.5
        step!(vec_env, actions)

    catch e
        @warn "Precompilation failure: $e"
    end
end

end