module RDE_Env
using CircularArrays: CircularArrays
using DrillInterface: DrillInterface
using DomainSets: DomainSets
using LinearAlgebra: LinearAlgebra, norm
using Logging: Logging, @logmsg, LogLevel
using Makie:
    Makie,
    @lift,
    Auto,
    Axis,
    Button,
    Colorbar,
    Figure,
    GridLayout,
    Keyboard,
    Label,
    Point2f,
    Relative,
    SliderGrid,
    Toggle,
    autolimits!,
    axislegend,
    barplot!,
    colsize!,
    events,
    heatmap!,
    hidespines!,
    hidexdecorations!,
    hideydecorations!,
    hlines!,
    ispressed,
    lift,
    lines!,
    linkxaxes!,
    linkyaxes!,
    record,
    resize_to_layout!,
    rowsize!,
    scatter!,
    scatterlines!,
    set_close_to!,
    stairs!,
    to_value,
    xlims!,
    ylims!,
    @L_str
using Observables: Observable, Observables, connect!, on
using OrdinaryDiffEq: ODEProblem, OrdinaryDiffEq, ReturnCode, SciMLBase
using PrecompileTools: @compile_workload, PrecompileTools
using ProgressMeter: Progress, next!
using RDE: AbstractControlShift, AbstractMethod, AbstractReset, RDE, RDEParam, RDEProblem,
    RDE_RHS!, set_spatial_control_smoothing!
using Random: Random
using Statistics: Statistics, mean
# using Polyester

include("core.jl")
export AbstractActionStrategy, AbstractMultiAgentObservationStrategy,
    AbstractObservationStrategy, AbstractRDEEnv, AbstractRewardStrategy,
    AbstractScalarActionStrategy, AbstractScalarRewardStrategy,
    AbstractVectorActionStrategy, AbstractVectorRewardStrategy
export compute_observation!
export _observation_space, apply_action!, on_reset!, set_reward!
export RDEEnv, RDEEnvCache
export AbstractCache, NoCache, initialize_cache, reset_cache!
export AbstractContextStrategy
include("utils.jl")
export get_avg_wave_speed, get_plotting_speed_adjustments

# Context
include("contexts.jl")
export NoContextStrategy

# Actions
include("actions/actions.jl")
export DirectScalarPressureAction, DirectVectorPressureAction, momentum

# Control shift strategies
include("control_shift.jl")
export MovingFrameControlShift

# Observation strategies
include("observations/observations.jl")
export FullStateCenteredObservation, FullStateObservation

# Rewards
include("rewards/rewards.jl")
export USpanReward
export compute_reward, set_reward!, set_termination_reward!
export ScalarToVectorReward

# Environment
include("RLenv.jl")
export _act!, _observe, _reset!

include("dril_interface.jl")
export MultiAgentRDEEnv, _action_space, _multi_agent_action_space

include("policies.jl")
export AbstractRDEPolicy, get_env
export PolicyRunData, run_policy
export _predict_action
export section_midpoint_indices, section_midpoint_values

# Plotting
include("plotting.jl")
export animate_policy, animate_policy_data, plot_policy, plot_policy_data,
    plot_shifted_history, plot_shifted_history!

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

        policy = RandomPolicy(env)
        data = run_policy(policy, env)
    catch e
        rethrow(e)
    end
end

end
