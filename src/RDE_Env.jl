module RDE_Env
    using RDE
    using CommonRLInterface
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
    using POMDPs
    using POMDPTools
    using PrecompileTools
    using ProgressMeter

    include("types.jl")
    # Actions
    include("actions.jl")
    export AbstractActionType, ScalarPressureAction, ScalarAreaScalarPressureAction, VectorPressureAction
    export action_dim, get_standard_normalized_actions

    # Observation strategies
    include("observations.jl")
    include("composite_observation.jl")
    include("multisection_observation.jl")
    export AbstractObservationStrategy, FourierObservation, StateObservation, SampledStateObservation, MultiSectionObservation
    export CompositeObservation
    export compute_observation, get_init_observation

   

    # Rewards
    include("rewards.jl")
    include("composite_reward.jl")
    include("multisection_reward.jl")
    export AbstractRDEReward, ShockSpanReward, ShockPreservingReward, ShockPreservingSymmetryReward
    export CompositeReward, ConstantTargetReward, MultiSectionReward
    export set_reward!, set_termination_reward!

    # Environment
    include("RLenv.jl")
    export RDEEnv, RDEEnvCache
    export reset!, act!, observe, state, terminated

    include("policies.jl")
    export Policy, StepwiseRDEPolicy, RandomRDEPolicy, ConstantRDEPolicy, SinusoidalRDEPolicy

    # Vectorized environments
    include("vec_env.jl")
    include("multi_agent_vec_env.jl")
    export RDEVecEnv, MultiAgentRDEVecEnv
    export step!, reset!, seed!

    # Policies
    export PolicyRunData, run_policy
    export ConstantRDEPolicy, SinusoidalRDEPolicy, StepwiseRDEPolicy, RandomRDEPolicy

    # Plotting
    include("plotting.jl")
    export plot_policy_data, plot_shifted_history, plot_policy, animate_policy, animate_policy_data

    include("interactive_control.jl")
    export interactive_control

    @compile_workload begin
        try
            # Create and run a small environment with random policy
            env = RDEEnv(;
                dt=0.01,
                params=RDEParam(;N=512, tmax=0.05),
                τ_smooth=0.001,
                momentum=0.8,
                observation_strategy=FourierObservation(8),
                action_type=ScalarPressureAction()
            )
            policy = RandomRDEPolicy(env)
            data = run_policy(policy, env, saves_per_action=2)
            
            # Test vectorized environment
            envs = [RDEEnv(;
                dt=0.01,
                τ_smooth=0.001,
                params=RDEParam(;N=512, tmax=0.05),
                observation_strategy=FourierObservation(8)
            ) for _ in 1:2]
            vec_env = RDEVecEnv(envs)
            reset!(vec_env)
            actions = rand(Float32, 1, 2) .- 0.5
            step!(vec_env, actions)
            
        catch e
            @warn "Precompilation failure: $e"
        end
    end

end