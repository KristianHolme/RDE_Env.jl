using Colors: RGB, distinguishable_colors

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
    return plot_policy_data(data, env)
end

"""
    plot_policy_data(data::PolicyRunData, env::RDEEnv; kwargs...)

Create an interactive visualization of policy execution data.

# Arguments
- `data::PolicyRunData`: Data from policy execution
- `env::RDEEnv`: RDE environment

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
        data::PolicyRunData,
        env::RDEEnv;
        fig_size = (1000, 900),
        kwargs...
    )
    fig = Figure(; size = fig_size)
    layout = fig[1, 1] = GridLayout()
    plot_policy_data!(
        layout,
        data,
        env;
        kwargs...
    )
    resize_to_layout!(fig)
    return fig
end

function plot_policy_data!(
        layout::GridLayout,
        data::PolicyRunData,
        env::RDEEnv;
        time_idx = Observable(1),
        player_controls = true,
        rewards_and_shocks = true,
        energy_and_chamber_pressure = false,
        control_history = true,
        observations = false,
        live_control = false,
        max_jump_speed = 4.6f0,
        initial_moving_frame = true,
        kwargs...
    )
    action_ts = data.action_ts
    ss = data.ss
    u_ps = data.u_ps
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

    # Frame shifting setup
    dx = env.prob.x[2] - env.prob.x[1]
    has_moving_frame = !isempty(data.control_shifts) && data.control_shifts[1] isa MovingFrameControlShift

    # Compute plotting frame speed vector (c) - same as plot_shifted_history
    us, = RDE.split_sol(states)
    c_vector = get_plotting_speed_adjustments(data, dx, max_speed = max_jump_speed)

    # Precompute cumulative positions for plotting frame (lab_frame -> plot_frame)
    plot_frame_pos = [zero(eltype(c_vector)); cumsum(c_vector .* diff(state_ts))]

    # Precompute control shift positions if using MovingFrameControlShift
    control_shift_positions = if has_moving_frame
        control_ts = getproperty.(data.control_shifts, :t_last)
        map(enumerate(state_ts)) do (i, t)
            control_ix = findlast(ct -> ct <= t, control_ts)
            isnothing(control_ix) && return zero(eltype(state_ts))
            RDE.get_control_shift(data.control_shifts[control_ix], Float32[], t)
        end
    else
        zeros(eltype(state_ts), length(state_ts))
    end

    # Observable shifts lifted on time_idx
    control_to_lab_shift = @lift(Int(round(control_shift_positions[$time_idx] / dx)))
    lab_to_plot_shift = @lift(Int(round(plot_frame_pos[$time_idx] / dx)))

    # Toggle for moving frame visualization
    moving_frame_toggle = Observable(initial_moving_frame)

    # Combined shift: 0 if toggle off, lab_to_plot_shift if toggle on
    plot_shift = @lift($moving_frame_toggle ? $lab_to_plot_shift : 0)

    # Raw data observables (u and λ are in lab frame from solver)
    raw_u = @lift(states[$time_idx][1:N])
    raw_λ = @lift(states[$time_idx][(N + 1):end])

    # Final shifted data: lab_frame -> plot_frame when toggle active
    u_data = @lift(circshift($raw_u, -$plot_shift))
    λ_data = @lift(circshift($raw_λ, -$plot_shift))
    # @info sparse_to_dense_ind(ts, sparse_ts, 3)
    # s = @lift(ss[sparse_to_dense_ind(state_ts, action_ts, $time_idx)])
    # u_p = @lift(u_ps[sparse_to_dense_ind(state_ts, action_ts, $time_idx)])

    main_layout = layout[1, 1] = GridLayout()
    upper_area = main_layout[1, 1] = GridLayout() #title, time
    system_plot_layout = main_layout[2, 1] = GridLayout()
    metrics_action_layout = nothing
    if control_history || energy_and_chamber_pressure || rewards_and_shocks
        metrics_action_layout = layout[1, 2] = GridLayout()
    end

    if control_history || energy_and_chamber_pressure || rewards_and_shocks
        colsize!(layout, 1, Relative(0.66))
    end


    label = Label(
        upper_area[1, 1],
        text = @lift(
            "Time: $(round(state_ts[$time_idx], digits = 1))" *
                ($moving_frame_toggle ? " [Moving Frame]" : "")
        ),
        tellwidth = false
    )

    RDE.main_plotting(
        system_plot_layout, env.prob.x, u_data, λ_data,
        env.prob.params;
        u_max = Observable(3),
        include_subfunctions = false,
        kwargs...
    )


    fine_time = @lift(state_ts[$time_idx])
    sparse_time_idx = @lift(dense_to_sparse_ind(state_ts, action_ts, $time_idx))
    sparse_time = @lift(action_ts[$sparse_time_idx])

    metrics_action_layout_plots = 0

    # energy_and_chamber_pressure keyword kept for backwards compatibility but ignored
    # energy_bal and chamber_p removed from PolicyRunData
    #     if energy_and_chamber_pressure
    #         metrics_action_Area_plots += 1
    #         ax_eb = Axis(metrics_action_area[1, metrics_action_Area_plots], title = "Energy balance", ylabel = "Ė")
    #         hidexdecorations!(ax_eb, grid = false)
    #         lines!(ax_eb, state_ts, energy_bal)
    #         vlines!(ax_eb, fine_time, color = :green, alpha = 0.5)
    #
    #         # Add chamber pressure
    #         ax_cp = Axis(metrics_action_area[2, metrics_action_Area_plots], title = "Chamber pressure", xlabel = "t", ylabel = "̄u²")
    #         lines!(ax_cp, state_ts, chamber_p)
    #         vlines!(ax_cp, fine_time, color = :green, alpha = 0.5)
    #     end

    if rewards_and_shocks
        metrics_action_layout_plots += 1
        # Add rewards and shocks
        layout = metrics_action_layout::GridLayout
        ax_rewards = Axis(layout[metrics_action_layout_plots, 1], title = "Rewards", ylabel = "r")
        hidexdecorations!(ax_rewards, grid = false)
        reward_color = :orange

        # Create reward times: first reward at second action_ts, last at end
        reward_times = if length(action_ts) > 1
            [action_ts[2:end]; state_ts[end]]
        else
            [action_ts[1] + 0.1]  # fallback if only one action
        end

        if eltype(rewards) <: AbstractVector
            # Plot all reward components as scatter points
            reward_lines = eachrow(stack(rewards))
            bg_color = RGB(ax_rewards.scene.backgroundcolor[])
            reward_colors = distinguishable_colors(length(rewards[1]), [bg_color]; dropseed = false)
            for i in 1:length(rewards[1])
                scatterlines!(ax_rewards, reward_times, reward_lines[i], color = reward_colors[i])
            end
            # Highlight current reward point (highlight reward earned at current time)
            for i in 1:length(rewards[1])
                scatter!(ax_rewards, @lift(reward_times[max(1, $sparse_time_idx - 1)]), @lift(rewards[max(1, $sparse_time_idx - 1)][i]), color = reward_colors[i], markersize = 12)
            end
        else
            # Plot rewards as scatter points
            scatterlines!(ax_rewards, reward_times, rewards, color = reward_color)
            # Highlight current reward point (highlight reward earned at current time)
            scatter!(ax_rewards, @lift(reward_times[max(1, $sparse_time_idx - 1)]), @lift(rewards[max(1, $sparse_time_idx - 1)]), color = reward_color, markersize = 12)
        end
        # vlines!(ax_rewards, fine_time, color=:green, alpha=0.5)
        metrics_action_layout_plots += 1
        layout = metrics_action_layout::GridLayout
        ax_shocks = Axis(layout[metrics_action_layout_plots, 1], title = "Shocks", xlabel = "t")
        dx = env.prob.x[2] - env.prob.x[1]
        us, = RDE.split_sol(states)
        shocks = RDE.count_shocks.(us, dx)
        lines!(ax_shocks, state_ts, shocks)
        scatter!(ax_shocks, fine_time, @lift(shocks[$time_idx]))
        # vlines!(ax_shocks, fine_time, color=:green, alpha=0.5)

        # Link x-axes of rewards and shocks plots
        linkxaxes!(ax_rewards, ax_shocks)
    end

    if control_history
        metrics_action_layout_plots += 1
        plot_s = !(norm(diff(ss)) ≈ 0)

        layout = metrics_action_layout::GridLayout
        ax_u_p = Axis(layout[metrics_action_layout_plots, 1], ylabel = "uₚ", yticklabelcolor = :royalblue)
        ylims!(ax_u_p, (0, env.u_pmax * 1.05))

        ax_s = nothing
        if plot_s
            metrics_action_layout_plots += 1
            ax_s = Axis(layout[metrics_action_layout_plots, 1], xlabel = "t", ylabel = "s", yticklabelcolor = :forestgreen, yaxisposition = :right)
            hidespines!(ax_s)
            hidexdecorations!(ax_s)
            hideydecorations!(ax_s, ticklabels = false, ticks = false, label = false)
        end

        if eltype(ss) <: AbstractVector && plot_s
            ax_s_local = ax_s::Axis
            lines!.(Ref(ax_s_local), Ref(action_ts), eachrow(stack(ss)), color = :forestgreen)
        elseif plot_s
            ax_s_local = ax_s::Axis
            lines!(ax_s_local, action_ts, ss, color = :forestgreen)
        end

        if eltype(u_ps) <: AbstractVector
            stairs!.(Ref(ax_u_p), Ref(action_ts), eachrow(stack(u_ps)), color = :royalblue, step = :post)
            for i in 1:length(u_ps[sparse_time_idx[]])
                lines!(ax_u_p, [action_ts[end], state_ts[end]], [u_ps[end][i], u_ps[end][i]], color = :royalblue)
                scatter!(ax_u_p, fine_time, @lift(u_ps[min($sparse_time_idx, length(u_ps))][i]), color = :royalblue)
            end
        else
            stairs!(ax_u_p, action_ts, u_ps, color = :royalblue, step = :post)
            lines!(ax_u_p, [action_ts[end], state_ts[end]], [u_ps[end], u_ps[end]], color = :royalblue)
            scatter!(ax_u_p, fine_time, @lift(u_ps[min($sparse_time_idx, length(u_ps))]), color = :royalblue)
        end

        #Time indicator
        # vlines!(ax_u_p, fine_time, color=:darkgreen, alpha=0.5)
    end

    # @show length(sparse_ts)
    if player_controls
        play_ctrl_area = main_layout[3, 1] = GridLayout()
        RDE.plot_controls(play_ctrl_area, time_idx, length(state_ts))

        toggle_layout = play_ctrl_area[3, 1] = GridLayout()
        # Add moving frame toggle (row 3, separate from speed controls in row 1-2)
        frame_toggle_label = Label(toggle_layout[1, 1], text = "Frame:", halign = :right, tellwidth = false)
        frame_toggle = Toggle(toggle_layout[1, 2], active = to_value(moving_frame_toggle), tellwidth = false)
        frame_toggle_text = Label(
            toggle_layout[1, 3],
            text = lift(a -> a ? "Moving" : "Lab", frame_toggle.active),
            halign = :left,
            tellwidth = false
        )
        connect!(moving_frame_toggle, frame_toggle.active)
    end

    if live_control
        max_u_p = eltype(u_ps) <: AbstractVector ? maximum(maximum.(u_ps)) : maximum(u_ps)
        ax_live_u_p = Axis(
            system_plot_layout[1, 1][1, 1][3, 1], ylabel = "uₚ", yaxisposition = :left,
            limits = ((nothing, (-0.1, max(max_u_p * 1.1, 1.0e-3))))
        )
        if eltype(u_ps) <: AbstractVector
            sections = env.action_strat.n_sections
            section_size = N ÷ sections

            # Raw u_p upsampled to full grid
            raw_u_p_upsampled = @lift begin
                raw_u_p = u_ps[min(length(u_ps), $sparse_time_idx + 1)]
                reduce(vcat, [fill(raw_u_p[i], section_size) for i in 1:length(raw_u_p)])
            end

            # Two-stage shifting using Observable shifts:
            # Stage 1: control_frame -> lab_frame (always if MovingFrameControlShift)
            u_p_in_lab_frame = @lift(circshift($raw_u_p_upsampled, $control_to_lab_shift))
            # Stage 2: lab_frame -> plot_frame (when toggle active, plot_shift is 0 otherwise)
            u_p_t = @lift(circshift($u_p_in_lab_frame, -$plot_shift))


            # Full grid x coordinates (always upsampled now)
            u_p_pts = collect(1:N) / N * L
            lines!(ax_live_u_p, u_p_pts, u_p_t)
        else
            stairs!(ax_live_u_p, action_ts, u_ps, color = :royalblue, step = :post)
            lines!(ax_live_u_p, [action_ts[end], state_ts[end]], [u_ps[end], u_ps[end]], color = :royalblue)
            scatter!(ax_live_u_p, fine_time, @lift(u_ps[min($sparse_time_idx, length(u_ps))]), color = :royalblue)
        end
    end

    if observations
        observation = @lift(data.observations[$sparse_time_idx])
        # @show observation[]
        if typeof(data.observations[1]) <: AbstractVector
            ax_obs = Axis(layout[end + 1, 1], title = "Observations")
            barplot!(ax_obs, observation)
        else
            ax_obs = Axis(layout[end + 1, 1], title = "Observations", xlabel = "index", ylabel = "Agent")
            xs = 1:size(observation[], 1)
            ys = 1:size(observation[], 2)
            heatmap!(ax_obs, xs, ys, observation)
        end
    end
    return nothing
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
        ts::AbstractArray, c::Union{Real, AbstractArray} = 1.71
    )
    fig = Figure(size = size)
    layout = fig[1, 1] = GridLayout()
    plot_shifted_history!(
        layout,
        us,
        x,
        ts,
        c
    )
    resize_to_layout!(fig)
    return fig
end

function plot_shifted_history!(
        layout::GridLayout,
        us::AbstractArray, x::AbstractArray,
        ts::AbstractArray, c::Union{Real, AbstractArray} = 1.71;
        u_ps = nothing, rewards = nothing,
        target_shock_count = nothing,
        action_ts = ts,
        plot_shocks = true,
        title = nothing,
        u_hm_kwargs = (),
        u_ax_kwargs = (),
        control_shifts = nothing,
        u_p_follow_u = true,
        movingframe = true,
    )

    dx = x[2] - x[1]
    pre_check_ts!(ts)
    pre_check_ts!(action_ts)
    shifted_us = Array.(RDE.shift_inds(us, x, ts, c))

    ax = Axis(
        layout[1, 1]; title = "u(ψ, t)", xlabel = "t",
        ylabel = "ψ", yzoomlock = true, ypanlock = true,
        limits = (extrema(ts), extrema(x)), xautolimitmargin = (0.0, 0.0),
        u_ax_kwargs...
    )
    hm = heatmap!(ax, ts, x, stack(shifted_us)', colorscale = identity, u_hm_kwargs...)
    Colorbar(layout[1, 2], hm)
    if plot_shocks
        counts = RDE.count_shocks.(us, dx)
        ax2 = Axis(
            layout[end + 1, 1], xlabel = "t", ylabel = "Number of shocks",
            limits = (nothing, (-0.05, maximum(counts) * 1.05)),
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
        u_p_minimum = minimum(minimum.(u_ps))
        u_p_maximum = maximum(maximum.(u_ps))
        if eltype(u_ps) <: AbstractVector
            #TODO: factor out some utils here?
            # u_ps is a vector field - plot on same spatial grid as u
            ax3 = Axis(
                layout[end + 1, 1]; xlabel = "t", ylabel = "ψ",
                yzoomlock = true, ypanlock = true,
                limits = (extrema(ts), extrema(x)),
                xautolimitmargin = (0.0, 0.0),
                title = "uₚ(ψ, t)",
                u_ax_kwargs...
            )
            points_per_section = length(x) ÷ length(u_ps[1])

            # Upsample spatially: from coarse sections to fine grid (piecewise constant)
            spatially_upsampled = map(u_ps) do u_p_coarse
                # Each coarse value gets repeated points_per_section times
                reduce(vcat, [fill(u_p_coarse[i], points_per_section) for i in 1:length(u_p_coarse)])
            end

            # For each fine timestep, find corresponding action timestep and shift appropriately
            # This preserves the stationary nature of u_ps in the lab frame
            upsampled_u_ps = map(enumerate(ts)) do (i, t)
                # Find which action timestep this corresponds to (piecewise constant in time)
                action_idx = searchsortedlast(action_ts, t)
                action_idx = clamp(action_idx, 1, length(spatially_upsampled))
                spatially_upsampled[action_idx]
            end

            @debug "upsampled_u_ps[]: $(length(upsampled_u_ps))"

            # if we have a moving reference frame we shift u_p forwards to match u
            static_ref_u_ps = deepcopy.(upsampled_u_ps)
            if movingframe && !isnothing(control_shifts) && control_shifts[1] isa MovingFrameControlShift
                control_ts = getproperty.(control_shifts, :t_last)
                for i in eachindex(static_ref_u_ps)
                    t = ts[i]
                    control_ix = findlast(ct -> ct <= t, control_ts)
                    if isnothing(control_ix)
                        error("No control shift found for time $t, control_ts: $control_ts, i: $i, ts: $ts, action_ts: $action_ts")
                    end
                    shift_pos = RDE.get_control_shift(control_shifts[control_ix], x, t)
                    shift = Int(round(shift_pos / dx))
                    @debug "t: $(round(t, digits = 2)), control_ix: $control_ix, shift_pos: $(round(shift_pos, digits = 2)), shift: $shift"

                    # Apply shifts
                    if shift != 0
                        circshift!(static_ref_u_ps[i], upsampled_u_ps[i], shift)
                    end
                end
                c_at_action_ts = c[searchsortedlast.(Ref(ts), action_ts)]
                @debug "c | control_speeds="
                @debug display(hcat(c_at_action_ts, getproperty.(control_shifts, :velocity)))
            end

            # Now we shift u_p backwards, matching u
            if u_p_follow_u
                shifted_u_ps = Array.(RDE.shift_inds(static_ref_u_ps, x, ts, c))
            else
                shifted_u_ps = copy(static_ref_u_ps)
            end
            hm_u_ps = heatmap!(ax3, ts, x, stack(shifted_u_ps)')
            Colorbar(layout[end, 2], hm_u_ps)
            # lines!.(Ref(ax3), Ref(action_ts), eachrow(stack(u_ps)), color = :royalblue)
            linkyaxes!(ax, ax3)
        else
            ax3 = Axis(
                layout[end + 1, 1], xlabel = "t", ylabel = "uₚ",
                limits = (nothing, (0.0, max(u_p_maximum * 1.05, 1.2))),
                xautolimitmargin = (0.0, 0.0),
            )
            if !isempty(u_ax_kwargs) && haskey(u_ax_kwargs, :xticks)
                ax3.xticks = u_ax_kwargs[:xticks]
            end
            stairs!(ax3, action_ts, u_ps, color = :royalblue, step = :post)

            lines!(ax3, [action_ts[end], ts[end]], [u_ps[end], u_ps[end]], color = :royalblue)
        end
        linkxaxes!(ax, ax3)
    end
    if rewards !== nothing
        reward_ts = if length(action_ts) > 1
            [action_ts[2:end]; ts[end]]
        else
            [action_ts[1] + 0.1]  # fallback if only one action
        end
        rewards_minimum = minimum(minimum.(rewards))
        rewards_maximum = maximum(maximum.(rewards))
        ax4 = Axis(
            layout[end + 1, 1], xlabel = "t", ylabel = "Reward",
            limits = (nothing, (rewards_minimum - 0.05, rewards_maximum + 0.05)),
            xautolimitmargin = (0.0, 0.0)
        )
        if eltype(rewards) <: AbstractVector
            scatterlines!.(Ref(ax4), Ref(reward_ts), eachrow(stack(rewards)), color = :orange)
        else
            scatterlines!(ax4, reward_ts, rewards, color = :orange)
        end
        linkxaxes!(ax, ax4)
    end
    autolimits!(ax)
    if title !== nothing
        Label(layout[0, 1], title, fontsize = 20, tellwidth = false)
    end
    return nothing
end

function plot_shifted_history(
        data::PolicyRunData,
        x::AbstractArray,
        c = :auto;
        use_rewards = true,
        max_jump_speed = 4.6f0,
        size = (1200, 600),
        kwargs...
    )
    us, = RDE.split_sol(data.states)
    saves_per_action = (length(data.state_ts) - 1) ÷ length(data.action_ts)
    if c == :auto
        c = get_plotting_speed_adjustments(data, x[2] - x[1], max_speed = max_jump_speed)
    elseif c == :legacy
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
    fig = Figure(size = size)
    layout = fig[1, 1] = GridLayout()
    plot_shifted_history!(
        layout,
        us,
        x,
        data.state_ts,
        c;
        u_ps = data.u_ps,
        rewards = use_rewards ? data.rewards : nothing,
        action_ts = data.action_ts,
        control_shifts = data.control_shifts,
        kwargs...
    )
    resize_to_layout!(fig)
    return fig
end

function animate_policy(π::P, env::RDEEnv; kwargs...) where {P <: AbstractRDEPolicy}
    data = run_policy(π, env)
    return animate_policy_data(data, env; kwargs...)
end

function animate_policy_data(
        data::PolicyRunData, env::RDEEnv;
        dir = "./videos/", fname = "policy", format = ".mp4", fps = 25, kwargs...
    )
    time_idx = Observable(1)
    time_steps = length(data.state_ts)
    fig = plot_policy_data(data, env; time_idx, player_controls = false, show_mouse_vlines = false, kwargs...)

    if !isdir(dir)
        mkpath(dir)
    end

    path = joinpath(dir, fname * format)
    p = Progress(time_steps, desc = "Recording animation...")
    return record(fig, path, 1:time_steps, framerate = fps) do i
        time_idx[] = i
        next!(p)
    end
end
