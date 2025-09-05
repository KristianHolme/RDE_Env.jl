"""
    get_timestep_scale(val)

Calculate an appropriate scale for timestep adjustments based on the current value.

# Arguments
- `val`: Current value to scale

# Returns
- Scaled step size (0.01 * val)
"""
function get_timestep_scale(val)
    return 0.01 * val
end

"""
    interactive_control(env::RDEEnv; callback=nothing, show_observations=false)

Create an interactive visualization and control interface for an RDE simulation.

# Keywords
- `callback`: Optional function called after each action with the environment as argument
- `show_observations`: Whether to show the observation plot (default: false)

# Returns
- `Tuple{RDEEnv, Figure}`: The RDE environment and Makie figure

# Controls
## Keyboard Controls
- `→`: Step simulation forward
- `↑`: Increase timestep
- `↓`: Decrease timestep
- `w`: Increase s parameter (ScalarAreaScalarPressureAction only)
- `s`: Decrease s parameter (ScalarAreaScalarPressureAction only)
- `e`: Increase u_p parameter or first section pressure
- `d`: Decrease u_p parameter or first section pressure
- `r`: Reset simulation

For VectorPressureAction:
- `e`/`d`: Modify pressure in all sections simultaneously
- (Individual sections can be controlled via sliders)

## Interactive Elements
- Sliders for control parameters based on action type:
  - ScalarAreaScalarPressureAction: s, u_p, and timestep sliders
  - ScalarPressureAction: u_p and timestep sliders  
  - VectorPressureAction: multiple u_p sliders (one per section) and timestep slider
- Control buttons:
  - Step: Advance simulation by one timestep
  - Reset: Reset simulation to initial state
- Real-time plots:
  - Velocity field (u)
  - Reaction progress (λ)
  - Reward
  - Observations (optional): barplot for vector observations, heatmap for matrix observations
- Time and parameter value displays

# Example
```julia
# Basic usage
env, fig = interactive_control()

# With observations displayed
env, fig = interactive_control(show_observations=true)

# Add custom callback
env, fig = interactive_control(callback=(env)->println("t = \$(env.t)"))

# With VectorPressureAction and MultiSectionObservation (matrix observations)
env = RDEEnv(action_type=VectorPressureAction(4), observation_strategy=MultiSectionObservation(4))
env, fig = interactive_control(env, show_observations=true)
```

# Notes
- The visualization updates in real-time as the simulation progresses
- Parameter changes are applied smoothly using the current timestep as the smoothing time
- The reward plot auto-scales as the simulation runs
- Automatically adapts interface based on the environment's action type
"""

function interactive_control(env::RDEEnv; callback = nothing, show_observations = false, dtmax = 1.0, kwargs...)
    params = env.prob.params
    N = params.N
    fig = Figure(size = (1200, show_observations ? 900 : 700))
    upper_area = fig[1, 1] = GridLayout()
    plotting_area = fig[2, 1] = GridLayout()
    energy_area = fig[3, 1][1, 1] = GridLayout()
    control_area = fig[3, 1][1, 2] = GridLayout()

    # Observables for real-time plotting
    u_data = Observable(env.state[1:N])
    λ_data = Observable(env.state[(N + 1):end])
    u_max = @lift(maximum($u_data))
    obs_data = show_observations ? Observable(_observe(env)) : nothing

    # energy_bal_pts = Observable(Point2f[(env.t, energy_balance(env.state, params))])
    # chamber_p_pts = Observable(Point2f[(env.t, chamber_pressure(env.state, params))])
    reward_pts = Observable(Point2f[(env.t, env.reward)])

    # Detect action type and create appropriate controls
    action_type = env.action_type

    # Always have timestep control
    time_step = Observable(env.dt)
    on(time_step) do val
        env.dt = val
        env.prob.method.cache.τ_smooth = val
    end

    # Initialize control observables and sliders based on action type
    if action_type isa ScalarAreaScalarPressureAction
        # Two controls: s and u_p
        control_s = Observable(params.s)
        s_start = params.s
        on(control_s) do val
            env.prob.method.cache.s_current .= val
        end

        control_u_p = Observable(params.u_p)
        u_p_start = params.u_p
        on(control_u_p) do val
            env.prob.method.cache.u_p_current .= val
        end

        # Create sliders using SliderGrid
        slider_grid = SliderGrid(
            control_area[1, 1],
            (label = "s", range = 0:0.001:env.smax, startvalue = control_s[]),
            (label = "u_p", range = 0:0.001:env.u_pmax, startvalue = control_u_p[]),
            (label = "Δt", range = 0:0.001:dtmax, startvalue = time_step[])
        )

        sliders = slider_grid.sliders
        slider_s = sliders[1]
        slider_u_p = sliders[2]
        slider_dt = sliders[3]

        on(slider_s.value) do val
            control_s[] = val
        end
        on(slider_u_p.value) do val
            control_u_p[] = val
        end
        on(slider_dt.value) do val
            time_step[] = val
        end

        # Labels
        s_label = slider_grid.labels[1]
        u_p_label = slider_grid.labels[2]
        dt_label = slider_grid.labels[3]

    elseif action_type isa ScalarPressureAction
        # Single control: u_p only
        control_u_p = Observable(params.u_p)
        u_p_start = params.u_p
        on(control_u_p) do val
            env.prob.method.cache.u_p_current .= val
        end

        # Create sliders using SliderGrid
        slider_grid = SliderGrid(
            control_area[1, 1],
            (label = "u_p", range = 0:0.001:env.u_pmax, startvalue = control_u_p[]),
            (label = "Δt", range = 0:0.001:dtmax, startvalue = time_step[])
        )

        sliders = slider_grid.sliders
        slider_u_p = sliders[1]
        slider_dt = sliders[2]

        on(slider_u_p.value) do val
            control_u_p[] = val
        end
        on(slider_dt.value) do val
            time_step[] = val
        end

        # Labels
        u_p_label = slider_grid.labels[1]
        dt_label = slider_grid.labels[2]

    elseif action_type isa VectorPressureAction
        # Multiple pressure controls: one per section
        n_sections = action_type.n_sections

        # Create observables for each section
        control_u_p_sections = [Observable(params.u_p) for _ in 1:n_sections]
        u_p_start = params.u_p

        # Set up listeners to update the cache when any section changes
        function update_cache()
            points_per_section = N ÷ n_sections
            for i in 1:n_sections
                start_idx = (i - 1) * points_per_section + 1
                end_idx = i * points_per_section
                env.prob.method.cache.u_p_current[start_idx:end_idx] .= control_u_p_sections[i][]
            end
            return
        end

        for i in 1:n_sections
            on(control_u_p_sections[i]) do val
                update_cache()
            end
        end

        # Create slider configurations
        slider_configs = [(label = "u_p_$i", range = 0:0.001:env.u_pmax, startvalue = control_u_p_sections[i][]) for i in 1:n_sections]
        push!(slider_configs, (label = "Δt", range = 0:0.001:dtmax, startvalue = time_step[]))

        # Create SliderGrid with all section controls plus timestep
        slider_grid = SliderGrid(control_area[1, 1], slider_configs...)

        sliders = slider_grid.sliders
        slider_u_p_sections = sliders[1:n_sections]
        slider_dt = sliders[end]

        # Set up slider listeners
        for i in 1:n_sections
            on(slider_u_p_sections[i].value) do val
                control_u_p_sections[i][] = val
            end
        end
        on(slider_dt.value) do val
            time_step[] = val
        end

        # Labels
        u_p_labels = slider_grid.labels[1:n_sections]
        dt_label = slider_grid.labels[end]

    else
        error("Unsupported action type: $(typeof(action_type))")
    end

    # Add control buttons
    button_area = control_area[2, 1] = GridLayout()
    step_button = Button(button_area[1, 1], label = "Step", tellwidth = false)
    reset_button = Button(button_area[1, 2], label = "Reset", tellwidth = false)

    time = Observable(env.t)

    """
    Update all visualization observables with current environment state.
    """
    function update_observables!()
        u_data[] = env.state[1:N]
        λ_data[] = env.state[(N + 1):end]
        time[] = env.t
        return if show_observations
            obs_data[] = _observe(env)
        end
    end

    # Button event handlers
    on(step_button.clicks) do _
        try
            # Create dummy action with correct length for action type
            dummy_action = if action_type isa VectorPressureAction
                zeros(action_type.n_sections)  # Vector with correct length
            else
                [0.0]  # Single value for scalar actions
            end
            _act!(env, dummy_action) # cached values are already set
            # energy_bal_pts[] = push!(energy_bal_pts[], Point2f(env.t, energy_balance(env.state, params)))
            # chamber_p_pts[] = push!(chamber_p_pts[], Point2f(env.t, chamber_pressure(env.state, params)))
            reward_pts[] = push!(reward_pts[], Point2f(env.t, env.reward))
            update_observables!()
            if callback !== nothing
                @debug "calling callback"
                callback(env)
            end
        catch e
            @error "error taking action e=$e"
            rethrow(e)
        end
    end

    on(reset_button.clicks) do _
        _reset!(env)
        update_observables!()

        # Reset controls to initial values
        if action_type isa ScalarAreaScalarPressureAction
            control_s[] = s_start
            control_u_p[] = u_p_start
            set_close_to!(slider_s, control_s[])
            set_close_to!(slider_u_p, control_u_p[])
        elseif action_type isa ScalarPressureAction
            control_u_p[] = u_p_start
            set_close_to!(slider_u_p, control_u_p[])
        elseif action_type isa VectorPressureAction
            for i in 1:action_type.n_sections
                control_u_p_sections[i][] = u_p_start
                set_close_to!(slider_u_p_sections[i], control_u_p_sections[i][])
            end
        end

        # energy_bal_pts[] = Point2f[(env.t, energy_balance(env.state, params))]
        # chamber_p_pts[] = Point2f[(env.t, chamber_pressure(env.state, params))]
        reward_pts[] = Point2f[(env.t, env.reward)]
    end

    # Create main visualization
    RDE.main_plotting(
        plotting_area, env.prob.x, u_data, λ_data, env.prob.params;
        u_max = u_max,
        s = action_type isa ScalarAreaScalarPressureAction ? control_s : Observable(params.s),
        u_p = action_type isa ScalarPressureAction ? control_u_p :
            action_type isa ScalarAreaScalarPressureAction ? control_u_p :
            Observable(params.u_p),
        kwargs...
    )

    # Create observation plot if requested
    if show_observations
        obs_area = fig[4, 1] = GridLayout()

        # Check if observations are vectors or matrices (consistent with plot_policy_data)
        if typeof(_observe(env)) <: AbstractVector
            # Vector observations: use barplot
            ax_obs = Axis(obs_area[1, 1], title = "Observations")
            barplot!(ax_obs, obs_data)
            on(obs_data) do obs
                ylims!(ax_obs, (min(-1.0, minimum(obs)), max(1.0, maximum(obs))))
            end
        else
            # Matrix observations: use heatmap
            ax_obs = Axis(obs_area[1, 1], title = "Observations", xlabel = "index", ylabel = "Agent")
            heatmap!(ax_obs, 1:size(_observe(env), 1), 1:size(_observe(env), 2), obs_data)
        end
        colsize!(obs_area, 1, Auto(0.3))
    end

    # Energy balance plot with auto-scaling (commented out)
    # eb_start_xmax = 0.5
    # ax_eb = Axis(energy_area[1, 1], title="Energy balance", xlabel="t", ylabel=L"Ė", limits=(0, eb_start_xmax, nothing, nothing))
    # lines!(ax_eb, energy_bal_pts)
    # on(energy_bal_pts) do _
    #     y_vals = getindex.(energy_bal_pts[], 2)
    #     ylims!(ax_eb, (min(0, minimum(y_vals)), maximum(y_vals)))
    #     xlims!(ax_eb, (0, max(eb_start_xmax, energy_bal_pts[][end][1])))
    # end

    # Chamber pressure plot with auto-scaling (commented out)
    # cp_start_xmax = 0.5
    # ax_cp = Axis(energy_area[1, 2], title="Chamber Pressure", xlabel="t", ylabel=L"P_c", limits=(0, cp_start_xmax, nothing, nothing))
    # lines!(ax_cp, chamber_p_pts)
    # on(chamber_p_pts) do _
    #     y_vals = getindex.(chamber_p_pts[], 2)
    #     ylims!(ax_cp, (min(0, minimum(y_vals)), maximum(y_vals)))
    #     xlims!(ax_cp, (0, max(cp_start_xmax, chamber_p_pts[][end][1])))
    # end

    # Reward plot with auto-scaling
    reward_start_xmax = 0.5
    ax_reward = Axis(energy_area[1, 1], title = "Reward", xlabel = "t", ylabel = "r", limits = (0, reward_start_xmax, nothing, nothing))
    lines!(ax_reward, reward_pts)
    on(reward_pts) do _
        y_vals = getindex.(reward_pts[], 2)
        ylims!(ax_reward, (min(-0.1, minimum(y_vals)), max(1.15, maximum(y_vals))))
        xlims!(ax_reward, (0, max(0.1, reward_pts[][end][1] * 1.15)))
    end

    rowsize!(fig.layout, 3, Auto(0.3))

    # Keyboard control setup
    pressed_keys = Set{Keyboard.Button}()
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press
            push!(pressed_keys, event.key)
        elseif event.action == Keyboard.release
            delete!(pressed_keys, event.key)
        end
    end

    """
    Background task that continuously checks for pressed keys and executes corresponding actions.
    """
    function key_action_loop()
        while events(fig.scene).window_open[]
            if events(fig.scene).hasfocus[]
                for key in pressed_keys
                    if key == Keyboard.up
                        # Increase time step on arrow up
                        time_step[] += get_timestep_scale(time_step.val)
                        time_step[] = min(time_step[], dtmax)
                        set_close_to!(slider_dt, time_step[])
                    elseif key == Keyboard.down
                        # Decrease time step on arrow down
                        time_step[] -= get_timestep_scale(time_step.val)
                        time_step[] = max(time_step[], 0.001)
                        set_close_to!(slider_dt, time_step[])
                    elseif key == Keyboard.right
                        try
                            # Create dummy action with correct length for action type
                            dummy_action = if action_type isa VectorPressureAction
                                zeros(action_type.n_sections)  # Vector with correct length
                            else
                                [0.0]  # Single value for scalar actions
                            end
                            _act!(env, dummy_action) #cached values are already set
                            # energy_bal_pts[] = push!(energy_bal_pts[], Point2f(env.t, energy_balance(env.state, params)))
                            # chamber_p_pts[] = push!(chamber_p_pts[], Point2f(env.t, chamber_pressure(env.state, params)))
                            reward_pts[] = push!(reward_pts[], Point2f(env.t, env.reward))
                            update_observables!()
                            if callback !== nothing
                                @debug "calling callback"
                                callback(env)
                            end
                        catch e
                            @error "error taking action e=$e"
                            rethrow(e)
                        end
                    elseif action_type isa ScalarAreaScalarPressureAction
                        # Both s and u_p controls available
                        if key == Keyboard.w
                            change = get_timestep_scale(control_s.val)
                            control_s[] += change
                            set_close_to!(slider_s, control_s[])
                        elseif key == Keyboard.s
                            control_s[] -= get_timestep_scale(control_s.val)
                            set_close_to!(slider_s, control_s[])
                        elseif key == Keyboard.e
                            control_u_p[] += get_timestep_scale(control_u_p.val)
                            set_close_to!(slider_u_p, control_u_p[])
                        elseif key == Keyboard.d
                            control_u_p[] -= get_timestep_scale(control_u_p.val)
                            set_close_to!(slider_u_p, control_u_p[])
                        end
                    elseif action_type isa ScalarPressureAction
                        # Only u_p control available
                        if key == Keyboard.e
                            control_u_p[] += get_timestep_scale(control_u_p.val)
                            set_close_to!(slider_u_p, control_u_p[])
                        elseif key == Keyboard.d
                            control_u_p[] -= get_timestep_scale(control_u_p.val)
                            set_close_to!(slider_u_p, control_u_p[])
                        end
                    elseif action_type isa VectorPressureAction
                        # All sections controlled simultaneously with e/d keys
                        if key == Keyboard.e
                            # Increase pressure in all sections
                            for i in 1:action_type.n_sections
                                control_u_p_sections[i][] += get_timestep_scale(control_u_p_sections[i].val)
                                set_close_to!(slider_u_p_sections[i], control_u_p_sections[i][])
                            end
                        elseif key == Keyboard.d
                            # Decrease pressure in all sections
                            for i in 1:action_type.n_sections
                                control_u_p_sections[i][] -= get_timestep_scale(control_u_p_sections[i].val)
                                set_close_to!(slider_u_p_sections[i], control_u_p_sections[i][])
                            end
                        end
                    end

                    if key == Keyboard.r
                        _reset!(env)
                        update_observables!()

                        # Reset controls to initial values
                        if action_type isa ScalarAreaScalarPressureAction
                            control_s[] = s_start
                            control_u_p[] = u_p_start
                            set_close_to!(slider_s, control_s[])
                            set_close_to!(slider_u_p, control_u_p[])
                        elseif action_type isa ScalarPressureAction
                            control_u_p[] = u_p_start
                            set_close_to!(slider_u_p, control_u_p[])
                        elseif action_type isa VectorPressureAction
                            for i in 1:action_type.n_sections
                                control_u_p_sections[i][] = u_p_start
                                set_close_to!(slider_u_p_sections[i], control_u_p_sections[i][])
                            end
                        end

                        # energy_bal_pts[] = Point2f[(env.t, energy_balance(env.state, params))]
                        # chamber_p_pts[] = Point2f[(env.t, chamber_pressure(env.state, params))]
                        reward_pts[] = Point2f[(env.t, env.reward)]
                    end
                end
            end
            sleep(0.1)  # Control loop rate
        end
        return
    end

    # Time label
    label = Label(upper_area[1, 1], text = @lift("Time: $(round($time, digits = 2))"), tellwidth = false)

    display(fig)
    @async key_action_loop()
    return env, fig
end
