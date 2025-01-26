using Makie
using RDE

"""
    plot_policy(π::Policy, env::RDEEnv)

Plot the results of running a policy in an RDE environment.

# Arguments
- `π::Policy`: Policy to evaluate
- `env::RDEEnv`: RDE environment

# Returns
- `Figure`: Makie figure containing the visualization
"""
function plot_policy(π::Policy, env::RDEEnv)
    data = run_policy(π, env)
    plot_policy_data(env, data)
end

"""
    plot_policy_data(env::RDEEnv, data::PolicyRunData; kwargs...)

Create an interactive visualization of policy execution data.

# Arguments
- `env::RDEEnv`: RDE environment
- `data::PolicyRunData`: Data from policy execution

# Keywords
- `time_idx::Observable{Int}=Observable(1)`: Observable for current time index
- `player_controls::Bool=true`: Whether to include playback controls

# Returns
- `Figure`: Makie figure containing the visualization

The plot includes:
- Velocity and reaction progress fields
- Energy balance
- Chamber pressure
- Rewards
- Number of shocks
- Control parameters (s and u_p)
- Interactive time controls (if enabled)
"""
function plot_policy_data(env::RDEEnv, data::PolicyRunData; 
        time_idx = Observable(1),
        player_controls=true,
        kwargs...)
    action_ts = data.action_ts
    ss = data.ss
    u_ps = data.u_ps
    energy_bal = data.energy_bal
    chamber_p = data.chamber_p
    rewards = data.rewards

    state_ts = data.state_ts
    states = data.states

    N = env.prob.params.N


    function sparse_to_dense_ind(dense_time::Vector, sparse_time::Vector, dense_ind::Int)
        dense_val = dense_time[dense_ind]
        sparse_ind = argmin(abs.(sparse_time .- dense_val))
        # @info sparse_ind dense_ind sp_val "$(dense_time[dense_ind])"
        return sparse_ind
    end


    
    u_data = @lift(states[$time_idx][1:N])
    λ_data = @lift(states[$time_idx][N+1:end])
    # @info sparse_to_dense_ind(ts, sparse_ts, 3)
    # s = @lift(ss[sparse_to_dense_ind(state_ts, action_ts, $time_idx)])
    # u_p = @lift(u_ps[sparse_to_dense_ind(state_ts, action_ts, $time_idx)])
    
    fig = Figure(size=(1000,900))
    upper_area = fig[1,1] = GridLayout()
    main_layout = fig[2,1] = GridLayout()
    metrics_action_area = fig[3,1] = GridLayout()
    

    rowsize!(fig.layout, 3, Auto(0.5))


    label = Label(upper_area[1,1], text=@lift("Time: $(round(state_ts[$time_idx], digits=2))"), tellwidth=false)

    RDE.main_plotting(main_layout, env.prob.x, u_data, λ_data, 
                env.prob.params;
                u_max = Observable(3),
                include_subfunctions = false,
                kwargs...)


    fine_time = @lift(state_ts[$time_idx])
    
    ax_eb = Axis(metrics_action_area[1,1], title="Energy balance", ylabel="Ė")
    hidexdecorations!(ax_eb, grid = false)
    lines!(ax_eb, state_ts, energy_bal)
    vlines!(ax_eb, fine_time, color=:green, alpha=0.5)

    # Add chamber pressure
    ax_cp = Axis(metrics_action_area[2,1], title="Chamber pressure", xlabel="t", ylabel="̄u²")
    lines!(ax_cp, state_ts, chamber_p)
    vlines!(ax_cp, fine_time, color=:green, alpha=0.5)

    # Add rewards and shocks
    ax_rewards = Axis(metrics_action_area[1,2], title="Rewards", ylabel="r")
    hidexdecorations!(ax_rewards, grid = false)
    lines!(ax_rewards, action_ts, rewards)
    vlines!(ax_rewards, fine_time, color=:green, alpha=0.5)

    ax_shocks = Axis(metrics_action_area[2,2], title="Shocks", xlabel="t")
    dx = env.prob.x[2] - env.prob.x[1]
    us, = RDE.split_sol(states)
    lines!(ax_shocks, state_ts, RDE.count_shocks.(us, dx))
    vlines!(ax_shocks, fine_time, color=:green, alpha=0.5)

    ax_s = Axis(metrics_action_area[1:2,3], xlabel="t", ylabel="s", yticklabelcolor=:forestgreen)
    ax_u_p = Axis(metrics_action_area[1:2,3], ylabel="u_p", yaxisposition = :right, yticklabelcolor=:royalblue)
    hidespines!(ax_u_p)
    hidexdecorations!(ax_u_p)
    hideydecorations!(ax_u_p, ticklabels=false, ticks=false, label=false)
    if ss isa Matrix
        lines!.(Ref(ax_s), Ref(action_ts), collect(eachrow(ss)), color=:forestgreen)
    else
        lines!(ax_s, action_ts, ss, color=:forestgreen)
    end
    if u_ps isa Matrix
        lines!.(Ref(ax_u_p), Ref(action_ts), collect(eachrow(u_ps)), color=:royalblue)
    else
        lines!(ax_u_p, action_ts, u_ps, color=:royalblue)
    end
    #Time indicator
    vlines!(ax_s, fine_time, color=:green, alpha=0.5)

    # @show length(sparse_ts)
    if player_controls
        play_ctrl_area = fig[4,1] = GridLayout()
        RDE.plot_controls(play_ctrl_area, time_idx, length(state_ts))
    end

    fig
end

"""
    plot_shifted_history(us::AbstractArray, x::AbstractArray, ts::AbstractArray, c::Union{Real, AbstractArray}; u_ps=nothing)

Create a space-time plot of the solution in a moving reference frame.

# Arguments
- `us::AbstractArray`: Array of velocity fields
- `x::AbstractArray`: Spatial grid points
- `ts::AbstractArray`: Time points
- `c::Union{Real, AbstractArray}`: Frame velocity (scalar or array)

# Keywords
- `u_ps=nothing`: Optional array of pressure values to plot

# Returns
- `Figure`: Makie figure containing:
  - Heatmap of velocity field in moving frame
  - Number of shocks over time
  - Pressure values over time (if provided)
"""
function plot_shifted_history(us::AbstractArray, x::AbstractArray,
         ts::AbstractArray, c::Union{Real, AbstractArray}; u_ps=nothing, rewards=nothing, target_shock_count=nothing)
    shifted_us = Array.(RDE.shift_inds(us, x, ts, c))

    fig = Figure(size=(1800, 600))
    ax = Axis(fig[1,1], title="u(x+ct, t)", xlabel="t",
            ylabel="x", yzoomlock=true, ypanlock=true,
            limits=(extrema(ts), extrema(x)), xautolimitmargin=(0.0, 0.0))
    hm = heatmap!(ax, ts, x, stack(shifted_us)', colorscale=identity)
    Colorbar(fig[1,2], hm)
    
    counts = RDE.count_shocks.(us, x[2] - x[1])
    ax2 = Axis(fig[2,1], xlabel="t", ylabel="Number of shocks", 
                limits=(nothing, (-0.05, maximum(counts)*1.05)),
                xautolimitmargin=(0.0, 0.0))
    lines!(ax2, ts, counts)
    if target_shock_count !== nothing
        hlines!(ax2, target_shock_count, color=:red, alpha=0.8, linestyle=:dash) #maybe change to stair if target shocks changes durin episode
    end
    linkxaxes!(ax, ax2)

    if u_ps !== nothing
        ax3 = Axis(fig[end+1,1], xlabel="t", ylabel="u_p", 
                    limits=(nothing, (minimum(u_ps)-0.05, maximum(u_ps)*1.05)),
                    xautolimitmargin=(0.0, 0.0))
        lines!(ax3, ts, u_ps)
        linkxaxes!(ax, ax3)
    end
    if rewards !== nothing
        ax4 = Axis(fig[end+1,1], xlabel="t", ylabel="Reward", 
                    limits=(nothing, (minimum(rewards)-0.05, maximum(rewards)*1.05)),
                    xautolimitmargin=(0.0, 0.0))
        lines!(ax4, ts, rewards)
        linkxaxes!(ax, ax4)
    end
    autolimits!(ax) 
    fig
end

function animate_policy(π::P, env::RDEEnv; kwargs...) where P <: Policy
    data = run_policy(π, env;)
    animate_policy_data(data, env; kwargs...)
end

function animate_policy_data(data::PolicyRunData, env::RDEEnv;
        dir_path="./videos/", fname="policy", format=".mp4", fps=25)
    time_idx = Observable(1)
    time_steps = length(data.state_ts)
    fig = plot_policy_data(env, data; time_idx, player_controls=false, show_mouse_vlines=false)

    if !isdir(dir_path)
        mkdir(dir_path)
    end

    path  = joinpath(dir_path, fname*format)
    p = Progress(time_steps, desc="Recording animation...");
    record(fig, path, 1:time_steps, framerate=fps) do i
        time_idx[] = i
        next!(p)
    end
end