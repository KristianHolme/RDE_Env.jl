using Colors: distinguishable_colors
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
- `e`: Increase u_p parameter by 0.01 or first section pressure
- `d`: Decrease u_p parameter by 0.01 or first section pressure
- `3`: Fine increase u_p parameter by 0.001
- `c`: Fine decrease u_p parameter by 0.001
- `r`: Reset simulation

For VectorPressureAction:
- `e`/`d`: Modify pressure in all sections by 0.01 simultaneously
- `3`/`c`: Fine control pressure in all sections by 0.001 simultaneously
- (Individual sections can be controlled via sliders)

## Interactive Elements
- Sliders for control parameters based on action type:
  - ScalarAreaScalarPressureAction: s (0.001 increments), u_p (0.001 increments), and timestep sliders
  - ScalarPressureAction: u_p (0.001 increments) and timestep sliders
  - VectorPressureAction: multiple u_p sliders (0.001 increments, one per section) and timestep slider
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
env = RDEEnv(action_strat=VectorPressureAction(4), observation_strat=MultiSectionObservation(4))
env, fig = interactive_control(env, show_observations=true)
```

# Notes
- The visualization updates in real-time as the simulation progresses
- Parameter changes are applied smoothly using the current timestep as the smoothing time
- The reward plot auto-scales as the simulation runs
- Automatically adapts interface based on the environment's action type
"""

function interactive_control(
        env::RDEEnv{T, A, O, R, V, OBS};
        callback = nothing, show_observations = false, dtmax = T(1.0), kwargs...
    ) where {T <: AbstractFloat, A, O, R, V, OBS}
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
    reward_value = env.reward
    reward_is_vector = reward_value isa AbstractVector || reward_value isa Tuple || reward_value isa NamedTuple

    function reward_components(reward_current)
        if reward_current isa AbstractVector
            return reward_current
        elseif reward_current isa Tuple
            return reward_current
        elseif reward_current isa NamedTuple
            return values(reward_current)
        else
            error("Unsupported reward type $(typeof(reward_current)) for vector reward plotting; expected AbstractVector, Tuple, or NamedTuple")
        end
    end

    reward_length = reward_is_vector ? length(reward_components(reward_value)) : 1
    reward_pts = if reward_is_vector
        comps = reward_components(reward_value)
        [Observable(Point2f[(env.t, comps[i])]) for i in 1:reward_length]
    else
        Observable(Point2f[(env.t, reward_value)])
    end

    function reset_reward_traces!(reward_current)
        if reward_is_vector
            comps = reward_components(reward_current)
            for (i, obs_pts) in enumerate(reward_pts)
                obs_pts[] = Point2f[(env.t, comps[i])]
            end
        else
            reward_pts[] = Point2f[(env.t, reward_current)]
        end
        return
    end

    function record_reward!(reward_current)
        if reward_is_vector
            comps = reward_components(reward_current)
            for (i, obs_pts) in enumerate(reward_pts)
                obs_pts[] = push!(obs_pts[], Point2f(env.t, comps[i]))
            end
        else
            reward_pts[] = push!(reward_pts[], Point2f(env.t, reward_current))
        end
        return
    end

    function reward_axis_bounds()
        if reward_is_vector
            y_vals = Float64[]
            t_max = 0.0
            for obs_pts in reward_pts
                data = obs_pts[]
                if !isempty(data)
                    append!(y_vals, Float64.(getindex.(data, 2)))
                    t_max = max(t_max, Float64(data[end][1]))
                end
            end
        else
            data = reward_pts[]
            y_vals = Float64.(getindex.(data, 2))
            t_max = Float64(data[end][1])
        end
        if isempty(y_vals)
            return (-0.1, 1.15, 0.1)
        end
        y_min = min(-0.1, minimum(y_vals))
        y_max = max(1.15, maximum(y_vals))
        x_max = max(0.1, t_max * 1.15)
        return (y_min, y_max, x_max)
    end

    # Detect action type and create appropriate controls (direct-only)
    action_strat = env.action_strat

    # Always have timestep control
    time_step = Observable(env.dt)
    on(time_step) do val
        env.dt = val
        env.prob.method.cache.τ_smooth = val
    end

    # Enforce supported action types and zero momentum
    if !(action_strat isa DirectScalarPressureAction || action_strat isa DirectVectorPressureAction)
        error("interactive_control supports only DirectScalarPressureAction or DirectVectorPressureAction")
    end

    # Action observables and sliders
    is_vector_action = action_strat isa DirectVectorPressureAction
    n_sections = is_vector_action ? action_strat.n_sections : 1

    # action_obs holds current action(s) in [-1, 1]
    init_action_value = mean(env.prob.method.cache.u_p_current)
    action_obs = is_vector_action ? Observable(fill(init_action_value, n_sections)) : Observable(init_action_value)

    # Slider grid: action slider(s) + dt
    minimum_action = T(0)
    maximum_action = T(1.2)
    if is_vector_action
        # Build per-section action sliders
        slider_configs = [(label = "u_p_$(i)", range = minimum_action:T(0.001):maximum_action, startvalue = action_obs[][i]) for i in 1:n_sections]
        push!(slider_configs, (label = "Δt", range = T(0):T(0.001):dtmax::T, startvalue = time_step[]))
        slider_grid = SliderGrid(control_area[1, 1], slider_configs...)
        sliders = slider_grid.sliders
        action_sliders = sliders[1:n_sections]
        slider_dt = sliders[end]

        # Keep action_obs and sliders in sync
        for i in 1:n_sections
            on(action_sliders[i].value) do val
                action_obs[][i] = clamp(val, minimum_action, maximum_action)
            end
        end
        on(slider_dt.value) do val
            time_step[] = val
        end
    else
        # Single scalar action slider
        slider_grid = SliderGrid(
            control_area[1, 1],
            (label = "u_p", range = minimum_action:T(0.001):maximum_action, startvalue = action_obs[]),
            (label = "Δt", range = T(0):T(0.001):dtmax::T, startvalue = time_step[])
        )
        sliders = slider_grid.sliders
        action_slider = sliders[1]
        slider_dt = sliders[2]

        on(action_slider.value) do val
            action_obs[] = clamp(val, minimum_action, maximum_action)
        end
        on(slider_dt.value) do val
            time_step[] = val
        end
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
            _act!(env, action_obs[])
            # energy_bal_pts[] = push!(energy_bal_pts[], Point2f(env.t, energy_balance(env.state, params)))
            # chamber_p_pts[] = push!(chamber_p_pts[], Point2f(env.t, chamber_pressure(env.state, params)))
            record_reward!(env.reward)
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

        if is_vector_action
            action_obs[] = fill(init_action_value, n_sections)
            for i in 1:n_sections
                set_close_to!(action_sliders[i], action_obs[][i])
            end
        else
            action_obs[] = init_action_value
            set_close_to!(action_slider, action_obs[])
        end

        # energy_bal_pts[] = Point2f[(env.t, energy_balance(env.state, params))]
        # chamber_p_pts[] = Point2f[(env.t, chamber_pressure(env.state, params))]
        reset_reward_traces!(env.reward)
    end

    # Create main visualization
    RDE.main_plotting(
        plotting_area, env.prob.x, u_data, λ_data, env.prob.params;
        u_max = u_max,
        s = action_strat isa ScalarAreaScalarPressureAction ? control_s : Observable(params.s),
        u_p = action_strat isa ScalarPressureAction ? control_u_p :
            action_strat isa ScalarAreaScalarPressureAction ? control_u_p :
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
            obs_hm = heatmap!(ax_obs, 1:size(_observe(env), 1), 1:size(_observe(env), 2), obs_data, colorrange = (0.0f0, 1.5f0))
            Colorbar(obs_area[1, 2], obs_hm)
        end
        colsize!(obs_area, 1, Auto(0.3))
    end

    # Reward plot with auto-scaling
    reward_start_xmax = 0.5
    ax_reward = Axis(energy_area[1, 1], title = "Reward", xlabel = "t", limits = (0, reward_start_xmax, nothing, nothing))
    if reward_is_vector
        bg_color = Makie.Colors.RGB(ax_reward.scene.backgroundcolor[])
        colors = distinguishable_colors(reward_length, [bg_color]; dropseed = true)
        labels = reward_value isa NamedTuple ? collect(keys(reward_value)) : ["r$(i)" for i in 1:reward_length]
        for i in 1:reward_length
            lines!(ax_reward, reward_pts[i], color = colors[i], label = labels[i])
        end
        axislegend(ax_reward)
        ax_reward.ylabel = "rᵢ"
    else
        lines!(ax_reward, reward_pts)
        ax_reward.ylabel = "r"
    end

    on(time) do _
        y_min, y_max, x_max = reward_axis_bounds()
        ylims!(ax_reward, (y_min, y_max))
        xlims!(ax_reward, (0, x_max))
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
                        time_step[] += 0.01
                        time_step[] = min(time_step[], dtmax)
                        set_close_to!(slider_dt, time_step[])
                    elseif key == Keyboard.down
                        # Decrease time step on arrow down
                        time_step[] -= 0.01
                        time_step[] = max(time_step[], 0.001)
                        set_close_to!(slider_dt, time_step[])
                    elseif key == Keyboard.right
                        try
                            _act!(env, action_obs[])
                            # energy_bal_pts[] = push!(energy_bal_pts[], Point2f(env.t, energy_balance(env.state, params)))
                            # chamber_p_pts[] = push!(chamber_p_pts[], Point2f(env.t, chamber_pressure(env.state, params)))
                            record_reward!(env.reward)
                            update_observables!()
                            if callback !== nothing
                                @debug "calling callback"
                                callback(env)
                            end
                        catch e
                            @error "error taking action" exception = (e, Base.catch_backtrace())
                            rethrow()  # preserve original backtrace
                        end
                    else
                        # Direct action types: adjust action observable(s)
                        if is_vector_action
                            if key == Keyboard.e || key == Keyboard.d || key == Keyboard._3 || key == Keyboard.c
                                delta = key == Keyboard._3 || key == Keyboard.c ? T(0.001) : T(0.01)
                                sign = (key == Keyboard.e || key == Keyboard._3) ? T(1) : T(-1)
                                a = clamp.(action_obs[] .+ sign * delta, minimum_action, maximum_action)
                                action_obs[] = a
                                # Sync sliders to action_obs
                                for i in 1:n_sections
                                    set_close_to!(action_sliders[i], action_obs[][i])
                                end
                            end
                        else
                            if key == Keyboard.e || key == Keyboard.d || key == Keyboard._3 || key == Keyboard.c
                                delta = key == Keyboard._3 || key == Keyboard.c ? T(0.001) : T(0.01)
                                sign = (key == Keyboard.e || key == Keyboard._3) ? T(1) : T(-1)
                                action_obs[] = clamp(action_obs[] + sign * delta, minimum_action, maximum_action)
                                set_close_to!(action_slider, action_obs[])
                            end
                        end
                    end

                    if key == Keyboard.r
                        _reset!(env)
                        update_observables!()


                        # energy_bal_pts[] = Point2f[(env.t, energy_balance(env.state, params))]
                        # chamber_p_pts[] = Point2f[(env.t, chamber_pressure(env.state, params))]
                        reset_reward_traces!(env.reward)
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
