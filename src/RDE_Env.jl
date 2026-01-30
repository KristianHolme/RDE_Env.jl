module RDE_Env
using CircularArrays: CircularArrays
using DRiL: DRiL, AbstractPolicy, terminated
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
    ylims!
using Observables: Observables, Observable, connect!, on
using OrdinaryDiffEq: OrdinaryDiffEq, ODEProblem, ReturnCode, SciMLBase
using PrecompileTools: PrecompileTools, @compile_workload
using ProgressMeter: ProgressMeter, Progress, next!
using RDE: RDE, AbstractControlShift, AbstractMethod, AbstractReset, RDEParam, RDEProblem, RDE_RHS!
using Random: Random
using Statistics: Statistics, mean
# using Polyester

include("core.jl")
export AbstractRDEEnv, AbstractActionStrategy, AbstractObservationStrategy, AbstractMultiAgentObservationStrategy,
    AbstractRewardStrategy, AbstractScalarActionStrategy, AbstractVectorActionStrategy,
    AbstractScalarRewardStrategy, AbstractVectorRewardStrategy
export compute_observation!
export set_reward!, apply_action!, on_reset!, _observation_space
export RDEEnv, RDEEnvCache
export AbstractCache, NoCache, initialize_cache, reset_cache!
export AbstractContextStrategy
include("utils.jl")
export get_plotting_speed_adjustments, get_avg_wave_speed

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
export MultiAgentRDEEnv, _action_space, _multi_agent_action_space

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
