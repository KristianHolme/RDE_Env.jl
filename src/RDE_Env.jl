module RDE_Env
using RDE
using CircularArrays
using DomainSets
using LinearAlgebra
using Statistics
using Random
using Logging
using SciMLBase
using OrdinaryDiffEq
using DRiL
using Makie
using Observables
# using Polyester
using PrecompileTools
using ProgressMeter

include("core.jl")
export AbstractRDEEnv, AbstractActionStrategy, AbstractObservationStrategy, AbstractMultiAgentObservationStrategy,
    AbstractRewardStrategy
export action_dim
export compute_observation!, get_init_observation
export set_reward!
export RDEEnv, RDEEnvCache
export AbstractCache, NoCache, initialize_cache, reset_cache!
export AbstractContextStrategy
include("utils.jl")
export get_plotting_speed_adjustments

# Context
include("contexts.jl")
export NoContextStrategy

# Actions
include("actions/actions.jl")
export DirectScalarPressureAction, DirectVectorPressureAction

# Control shift strategies
include("control_shift.jl")
export MovingFrameControlShift

# Observation strategies
include("observations/observations.jl")
export
    FullStateObservation, FullStateCenteredObservation

# Rewards
include("rewards/rewards.jl")
export USpanReward
export set_reward!, set_termination_reward!, compute_reward
export ScalarToVectorReward

# Environment
include("RLenv.jl")
export _reset!, _act!, _observe, terminated

include("dril_interface.jl")
export MultiAgentRDEEnv

include("policies.jl")
export AbstractRDEPolicy, get_env
export PolicyRunData, run_policy
export _predict_action

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
            observation_strat = FullStateObservation(),
            action_strat = DirectScalarPressureAction()
        )

        policy = DRiL.RandomPolicy(env)
        data = run_policy(policy, env)
    catch e
        rethrow(e)
    end
end

end
