using Makie
using RDE

"""
    plot_policy(π::AbstractRDEPolicy, env::RDEEnv)

Plot the results of running a policy in an RDE environment.

# Arguments
- `π::AbstractRDEPolicy`: Policy to evaluate
- `env::RDEEnv`: RDE environment

# Returns
- `Figure`: Makie figure containing the visualization
"""
function plot_policy(π::AbstractRDEPolicy, env::RDEEnv)
    data = run_policy(π, env)
    return plot_policy_data(env, data)
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
- `rewards_and_shocks::Bool=true`: Whether to show rewards and shock count plots
- `energy_and_chamber_pressure::Bool=false`: Whether to show energy balance and chamber pressure plots
- `control_history::Bool=true`: Whether to show control parameter history
- `observations::Bool=false`: Whether to show observation data
- `live_control::Bool=false`: Whether to enable live control interface
- `size::Tuple{Int,Int}=(1000,900)`: Size of the figure window

# Returns
- `Figure`: Makie figure containing the visualization

The plot includes:
- Velocity and reaction progress fields
- Optional plots based on keyword arguments:
  - Rewards and shock count
  - Energy balance and chamber pressure
  - Control parameter history (s and u_p)
  - Observation data visualization
  - Live control interface
- Interactive time controls (if player_controls=true)
"""
function plot_policy_data(
        env::RDEEnv, data::PolicyRunData;
        time_idx = Observable(1),
        player_controls = true,
        rewards_and_shocks = true,
        energy_and_chamber_pressure = false,
        control_history = true,
        observations = false,
        live_control = false,
        size = (1000, 900),
        kwargs...
    )
    action_ts = data.action_ts
    ss = data.ss
    u_ps = data.u_ps
    energy_bal = data.energy_bal
    chamber_p = data.chamber_p
    rewards = data.rewards

    state_ts = data.state_ts
    states = data.states

    N = env.prob.params.N
    L = env.prob.params.L


    function dense_to_sparse_ind(dense_time::Vector, sparse_time::Vector, dense_ind::Int)
        dense_val = dense_time[dense_ind]
        sparse_ind = findlast(x -> x <= dense_val, sparse_time)
        # sparse_ind = argmin(abs.(sparse_time .- dense_val))
        # @info sparse_ind dense_ind sp_val "$(dense_time[dense_ind])"
        return sparse_ind
    end


    u_data = @lift(states[$time_idx][1:N])
    λ_data = @lift(states[$time_idx][(N + 1):end])
    # @info sparse_to_dense_ind(ts, sparse_ts, 3)
    # s = @lift(ss[sparse_to_dense_ind(state_ts, action_ts, $time_idx)])
    # u_p = @lift(u_ps[sparse_to_dense_ind(state_ts, action_ts, $time_idx)])

    fig = Figure(; size)
    upper_area = fig[1, 1] = GridLayout()
    main_layout = fig[2, 1] = GridLayout()
    if control_history || energy_and_chamber_pressure || rewards_and_shocks
        metrics_action_area = fig[end + 1, 1] = GridLayout()
    end

    if control_history || energy_and_chamber_pressure || rewards_and_shocks
        rowsize!(fig.layout, 3, Auto(0.5))
    end


    label = Label(upper_area[1, 1], text = @lift("Time: $(round(state_ts[$time_idx], digits = 1))"), tellwidth = false)

    RDE.main_plotting(
        main_layout, env.prob.x, u_data, λ_data,
        env.prob.params;
        u_max = Observable(3),
        include_subfunctions = false,
        kwargs...
    )


    fine_time = @lift(state_ts[$time_idx])
    sparse_time_idx = @lift(dense_to_sparse_ind(state_ts, action_ts, $time_idx))
    sparse_time = @lift(action_ts[$sparse_time_idx])

    metrics_action_Area_plots = 0

    if energy_and_chamber_pressure
        metrics_action_Area_plots += 1
        ax_eb = Axis(metrics_action_area[1, metrics_action_Area_plots], title = "Energy balance", ylabel = "Ė")
        hidexdecorations!(ax_eb, grid = false)
        lines!(ax_eb, state_ts, energy_bal)
        vlines!(ax_eb, fine_time, color = :green, alpha = 0.5)

        # Add chamber pressure
        ax_cp = Axis(metrics_action_area[2, metrics_action_Area_plots], title = "Chamber pressure", xlabel = "t", ylabel = "̄u²")
        lines!(ax_cp, state_ts, chamber_p)
        vlines!(ax_cp, fine_time, color = :green, alpha = 0.5)
    end

    if rewards_and_shocks
        metrics_action_Area_plots += 1
        # Add rewards and shocks
        ax_rewards = Axis(metrics_action_area[1, metrics_action_Area_plots], title = "Rewards", ylabel = "r")
        hidexdecorations!(ax_rewards, grid = false)
        reward_color = :orange
        if eltype(rewards) <: AbstractVector
            stairs!.(Ref(ax_rewards), Ref(action_ts), eachrow(stack(rewards)), color = reward_color)
            for i in 1:length(rewards[sparse_time_idx[]])
                scatter!(ax_rewards, fine_time, @lift(rewards[min($sparse_time_idx + 1, length(rewards))][i]), color = reward_color)
            end
        else
            stairs!(ax_rewards, action_ts, rewards, color = reward_color)
            scatter!(ax_rewards, fine_time, @lift(rewards[min($sparse_time_idx + 1, length(rewards))]), color = reward_color)
        end
        # vlines!(ax_rewards, fine_time, color=:green, alpha=0.5)

        ax_shocks = Axis(metrics_action_area[2, metrics_action_Area_plots], title = "Shocks", xlabel = "t")
        dx = env.prob.x[2] - env.prob.x[1]
        us, = RDE.split_sol(states)
        shocks = RDE.count_shocks.(us, dx)
        lines!(ax_shocks, state_ts, shocks)
        scatter!(ax_shocks, fine_time, @lift(shocks[$time_idx]))
        # vlines!(ax_shocks, fine_time, color=:green, alpha=0.5)
    end

    if control_history
        metrics_action_Area_plots += 1
        plot_s = !(norm(diff(ss)) ≈ 0)

        ax_u_p = Axis(metrics_action_area[1:2, metrics_action_Area_plots], ylabel = "uₚ", yticklabelcolor = :royalblue)
        ylims!(ax_u_p, (0, env.u_pmax * 1.05))

        if plot_s
            ax_s = Axis(metrics_action_area[1:2, metrics_action_Area_plots], xlabel = "t", ylabel = "s", yticklabelcolor = :forestgreen, yaxisposition = :right)
            hidespines!(ax_s)
            hidexdecorations!(ax_s)
            hideydecorations!(ax_s, ticklabels = false, ticks = false, label = false)
        end

        if eltype(ss) <: AbstractVector && plot_s
            lines!.(Ref(ax_s), Ref(action_ts), eachrow(stack(ss)), color = :forestgreen)
        elseif plot_s
            lines!(ax_s, action_ts, ss, color = :forestgreen)
        end

        if eltype(u_ps) <: AbstractVector
            stairs!.(Ref(ax_u_p), Ref(action_ts), eachrow(stack(u_ps)), color = :royalblue)
            for i in 1:length(u_ps[sparse_time_idx[]])
                scatter!(ax_u_p, fine_time, @lift(u_ps[min($sparse_time_idx + 1, length(u_ps))][i]), color = :royalblue)
            end
        else
            stairs!(ax_u_p, action_ts, u_ps, color = :royalblue)
            scatter!(ax_u_p, fine_time, @lift(u_ps[min($sparse_time_idx + 1, length(u_ps))]), color = :royalblue)
        end

        #Time indicator
        # vlines!(ax_u_p, fine_time, color=:darkgreen, alpha=0.5)
    end

    # @show length(sparse_ts)
    if player_controls
        play_ctrl_area = fig[4, 1] = GridLayout()
        RDE.plot_controls(play_ctrl_area, time_idx, length(state_ts))
    end

    if live_control && eltype(u_ps) <: AbstractVector
        u_p_t = @lift(u_ps[$sparse_time_idx])
        max_u_p = RDE.turbo_maximum(RDE.turbo_maximum.(u_ps))
        ax_live_u_p = Axis(
            main_layout[1, 1][3, 1], ylabel = "uₚ", yaxisposition = :left,
            limits = ((nothing, (-0.1, max(max_u_p * 1.1, 1.0e-3))))
        )
        sections = env.action_type.n_sections
        section_size = N ÷ sections
        start = 1 + section_size ÷ 2
        u_p_pts = collect(start:section_size:N) / N * L
        stairs!(ax_live_u_p, u_p_pts, u_p_t, step = :center)
    end

    if observations
        observation = @lift(data.observations[$sparse_time_idx])
        # @show observation[]
        if typeof(data.observations[1]) <: AbstractVector
            ax_obs = Axis(fig[end + 1, 1], title = "Observations")
            barplot!(ax_obs, observation)
        else
            ax_obs = Axis(fig[end + 1, 1], title = "Observations", xlabel = "index", ylabel = "Agent")
            heatmap!(ax_obs, 1:size(observation[], 1), 1:size(observation[], 2), observation)
        end
    end
    resize_to_layout!(fig)
    return fig
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
function plot_shifted_history(
        us::AbstractArray, x::AbstractArray,
        ts::AbstractArray, c::Union{Real, AbstractArray} = 1.65;
        u_ps = nothing, rewards = nothing,
        target_shock_count = nothing,
        action_ts = ts,
        plot_shocks = true,
        title = nothing,
        size = (1200, 600)
    )
    pre_check_ts!(ts)
    pre_check_ts!(action_ts)
    shifted_us = Array.(RDE.shift_inds(us, x, ts, c))

    fig = Figure(size = size)
    ax = Axis(
        fig[1, 1], title = "u(ψ, t)", xlabel = "t",
        ylabel = "ψ", yzoomlock = true, ypanlock = true,
        limits = (extrema(ts), extrema(x)), xautolimitmargin = (0.0, 0.0)
    )
    hm = heatmap!(ax, ts, x, stack(shifted_us)', colorscale = identity)
    Colorbar(fig[1, 2], hm)
    if plot_shocks
        counts = RDE.count_shocks.(us, x[2] - x[1])
        ax2 = Axis(
            fig[end + 1, 1], xlabel = "t", ylabel = "Number of shocks",
            limits = (nothing, (-0.05, RDE.turbo_maximum(counts) * 1.05)),
            xautolimitmargin = (0.0, 0.0)
        )
        lines!(ax2, ts, counts)
        if target_shock_count !== nothing
            hlines!(ax2, target_shock_count, color = :red, alpha = 0.8, linestyle = :dash) #maybe change to stair if target shocks changes durin episode
            # Adjust y limits to ensure hline is visible
            ylims!(ax2, (ax2.limits[][2][1], max(ax2.limits[][2][2], target_shock_count * 1.05)))
        end
        linkxaxes!(ax, ax2)
    end

    if u_ps !== nothing
        u_p_minimum = RDE.turbo_minimum(RDE.turbo_minimum.(u_ps))
        u_p_maximum = RDE.turbo_maximum(RDE.turbo_maximum.(u_ps))
        ax3 = Axis(
            fig[end + 1, 1], xlabel = "t", ylabel = "uₚ",
            limits = (nothing, (0.0, max(u_p_maximum * 1.05, 1.2))),
            xautolimitmargin = (0.0, 0.0)
        )
        if eltype(u_ps) <: AbstractVector
            lines!.(Ref(ax3), Ref(action_ts), eachrow(stack(u_ps)), color = :royalblue)
        else
            lines!(ax3, action_ts, u_ps, color = :royalblue)
        end
        linkxaxes!(ax, ax3)
    end
    if rewards !== nothing
        rewards_minimum = RDE.turbo_minimum(RDE.turbo_minimum.(rewards))
        rewards_maximum = RDE.turbo_maximum(RDE.turbo_maximum.(rewards))
        ax4 = Axis(
            fig[end + 1, 1], xlabel = "t", ylabel = "Reward",
            limits = (nothing, (rewards_minimum - 0.05, rewards_maximum + 0.05)),
            xautolimitmargin = (0.0, 0.0)
        )
        if eltype(rewards) <: AbstractVector
            lines!.(Ref(ax4), Ref(action_ts), eachrow(stack(rewards)), color = :orange)
        else
            lines!(ax4, action_ts, rewards, color = :orange)
        end
        linkxaxes!(ax, ax4)
    end
    autolimits!(ax)
    if title !== nothing
        Label(fig[0, 1], title, fontsize = 20, tellwidth = false)
    end
    return fig
end

function plot_shifted_history(data::PolicyRunData, x::AbstractArray, c = :auto; use_rewards = true, kwargs...)
    us, = RDE.split_sol(data.states)
    saves_per_action = (length(data.state_ts) - 1) ÷ (length(data.action_ts) - 1)
    if c == :auto
        counts = RDE.count_shocks.(us, x[2] - x[1])
        u_ps = data.u_ps
        if eltype(u_ps) <: AbstractVector
            u_ps = mean.(u_ps)
        end
        if saves_per_action > 1
            u_ps = [u_ps[1]; repeat(u_ps[2:end], inner = saves_per_action)]
        end
        @assert length(u_ps) == length(counts) "length(u_ps) ($(length(u_ps))) != length(counts) ($(length(counts))), saves_per_action: $saves_per_action"
        speeds = RDE.predict_speed.(u_ps, counts)
        c = speeds[1:(end - 1)]
    end
    return plot_shifted_history(
        us, x, data.state_ts, c;
        u_ps = data.u_ps, rewards = use_rewards ? data.rewards : nothing, action_ts = data.action_ts, kwargs...
    )
end

function animate_policy(π::P, env::RDEEnv; kwargs...) where {P <: AbstractRDEPolicy}
    data = run_policy(π, env)
    return animate_policy_data(data, env; kwargs...)
end

function animate_policy_data(
        data::PolicyRunData, env::RDEEnv;
        dir_path = "./videos/", fname = "policy", format = ".mp4", fps = 25, kwargs...
    )
    time_idx = Observable(1)
    time_steps = length(data.state_ts)
    fig = plot_policy_data(env, data; time_idx, player_controls = false, show_mouse_vlines = false, kwargs...)

    if !isdir(dir_path)
        mkdir(dir_path)
    end

    path = joinpath(dir_path, fname * format)
    p = Progress(time_steps, desc = "Recording animation...")
    return record(fig, path, 1:time_steps, framerate = fps) do i
        time_idx[] = i
        next!(p)
    end
end
