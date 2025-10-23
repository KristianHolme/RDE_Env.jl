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
# using Polyester
# using POMDPTools
using PrecompileTools
using ProgressMeter

include("core.jl")
export AbstractRDEEnv, AbstractActionStrategy, AbstractObservationStrategy, AbstractMultiAgentObservationStrategy,
    AbstractRewardStrategy
export action_dim, _reset_action!, set_N!
export compute_observation, get_init_observation
export set_reward!
export RDEEnv, RDEEnvCache
export AbstractCache, NoCache, CompositeRewardCache, initialize_cache, reset_cache!
export get_target_shock_count, set_target_shock_count!
export AbstractGoalStrategy, AbstractGoalCache
include("utils.jl")
export sigmoid_to_linear, reward_sigmoid, sigmoid, linear_to_sigmoid, get_plotting_speed_adjustments

# goals
include("goals.jl")
export FixedTargetGoal, RandomTargetGoal, EvalCycleTargetGoal

# Actions
include("actions/actions.jl")
export ScalarPressureAction, ScalarAreaScalarPressureAction, VectorPressureAction, PIDAction
export LinearScalarPressureAction, LinearVectorPressureAction
export DirectScalarPressureAction, DirectVectorPressureAction

# Observation strategies
include("observations/observations.jl")
export
    FourierObservation, StateObservation,
    SampledStateObservation, MultiSectionObservation, SectionedStateObservation,
    MultiCenteredObservation, MeanInjectionPressureObservation,
    SectionedStateWithPressureHistoryObservation, MultiCenteredObservationWithPressureHistory
export CompositeObservation, compute_sectioned_observation

# Rewards
include("rewards/rewards.jl")
export ShockSpanReward, ShockPreservingReward, ShockPreservingSymmetryReward
export CompositeReward, ConstantTargetReward, MultiSectionReward, PeriodicityReward
export TimeAggCompositeReward, TimeMin, TimeAvg, TimeMax, TimeSum, TimeProd, TimeAggMultiSectionReward
export TimeDiffNormReward, MultiplicativeReward, PeriodMinimumReward, MultiSectionPeriodMinimumReward,
    PeriodMinimumVariationReward, ExponentialAverageReward, TransitionBasedReward, StabilityReward,
    StabilityTargetReward
export set_reward!, set_termination_reward!, compute_reward
export ScalarToVectorReward

# Environment
include("RLenv.jl")
export _reset!, _act!, _observe, state, terminated

include("policies.jl")
export AbstractRDEPolicy, StepwiseRDEPolicy, RandomRDEPolicy, ConstantRDEPolicy, SinusoidalRDEPolicy,
    DelayedPolicy, LinearPolicy, get_env, SawtoothPolicy, PIDControllerPolicy
export reset_pid_cache!, _predict_action
# Vectorized environments
# include("vec_env.jl")
# include("multi_agent_vec_env.jl")
# export RDEVecEnv, MultiAgentRDEVecEnv
# export step!, seed!
# export ThreadingMode, POLYESTER, THREADS

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
            dt = 0.1f0,
            params = RDEParam(; N = 512, tmax = 0.5f0),
            τ_smooth = 0.05f0,
            observation_strat = FourierObservation(8),
            action_strat = ScalarPressureAction()
        )
        policy = RandomRDEPolicy(env)
        data = run_policy(policy, env)
    catch e
        rethrow(e)
    end
end

end
