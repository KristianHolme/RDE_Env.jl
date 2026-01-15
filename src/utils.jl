function pre_check_ts!(ts::AbstractArray{T}) where {T <: AbstractFloat}
    return if ts[end] ≈ ts[end - 1]
        ts[end] += T(1.0e-6)
    end
end

function get_block_shock_data(shock_locations, blocks_ixs, block::Int)
    n_blocks = length(blocks_ixs) - 1
    @assert block in 1:n_blocks
    start_ix = blocks_ixs[block]
    end_ix = blocks_ixs[block + 1] - 1
    return shock_locations[start_ix:end_ix]
end

function get_block_shock_indices(block_shock_locations)
    n_shocks = sum(block_shock_locations[1])
    N = length(block_shock_locations[1])
    n_ts = length(block_shock_locations)
    shock_positions = [zeros(Int, n_ts) for _ in 1:n_shocks]
    start_positions = findall(block_shock_locations[1])
    for s_i in 1:n_shocks
        shock_positions[s_i][1] = start_positions[s_i]
    end
    for t_i in 2:n_ts
        for s_i in 1:n_shocks
            p_i = shock_positions[s_i][t_i - 1]
            dp = findfirst(block_shock_locations[t_i][p_i:(p_i + N - 1)])
            shock_positions[s_i][t_i] = p_i + dp
        end
    end
    return shock_positions
end


function get_block_speeds(block_shock_indices, dx::T, ts::AbstractVector{T}) where {T <: AbstractFloat}
    n_shocks = length(block_shock_indices)
    n_ts = length(block_shock_indices[1])
    @assert all(length.(block_shock_indices) .== n_ts)
    speeds = [zeros(T, n_ts) for _ in 1:n_shocks]
    for s_i in 1:n_shocks
        indices = block_shock_indices[s_i]
        for t_i in 1:(n_ts - 1)
            speeds[s_i][t_i] = (indices[t_i + 1] - indices[t_i]) * dx / (ts[t_i + 1] - ts[t_i])
        end
        speeds[s_i][n_ts] = speeds[s_i][n_ts - 1]
    end
    return speeds
end

function speed_tracking(data, dx::T) where {T <: AbstractFloat}
    # for each block of time having the same number of shocks, get the speed of each shock at each time point
    # for each block we have a vector of speed vectors, one for each shock
    # for blocks with no shocks or single point blocks, interpolate the speed from the neighboring blocks
    us, _ = RDE.split_sol(data.states)
    ts = data.state_ts
    return speed_tracking(us, ts, dx)
end

"""
    get_avg_wave_speed(us::Vector{Vector{T}}, ts, dx::T) -> T

Get the average wave speed from the solution data. Computes block speeds. If there is data, average speed for each shock,
then average the speeds of all shocks to get average speed for the block. Then average again to get average speed over all the blocks. 

# Arguments
- `us::Vector{Vector{T}}`: Solution data
- `ts`: Time points
- `dx::T`: Spatial spacing

"""
function get_avg_wave_speed(us::Vector{Vector{T}}, ts, dx::T) where {T <: AbstractFloat}

    shock_locations = RDE.shock_locations.(us, dx)
    shock_counts = sum.(shock_locations)
    blocks_ixs = [1, (findall(diff(shock_counts) .!= 0) .+ 1)..., length(shock_counts) + 1]
    n_blocks = length(blocks_ixs) - 1

    shock_speed_sum = zero(T)
    n_blocks_with_data = 0
    for block in 1:n_blocks
        block_start_ix = blocks_ixs[block]
        block_end_ix = blocks_ixs[block + 1] - 1
        block_shock_locations = get_block_shock_data(shock_locations, blocks_ixs, block)
        n_shocks = sum(block_shock_locations[1])


        if n_shocks == 0 || length(block_shock_locations) == 1
            continue
        end
        block_shock_indices = get_block_shock_indices(block_shock_locations)
        @assert length(block_shock_indices[1]) == length(block_shock_locations) "length(block_shock_indices[1]) ($(length(block_shock_indices[1]))) != length(block_shock_locations) ($(length(block_shock_locations)))"
        block_speeds = get_block_speeds(block_shock_indices, dx, ts[block_start_ix:block_end_ix])
        shocks_avg_speed = mean.(block_speeds)
        shock_speed_sum += mean(shocks_avg_speed)
        n_blocks_with_data += 1
    end
    if n_blocks_with_data == 0
        return -one(T)
    else
        return shock_speed_sum / n_blocks_with_data
    end
end

function speed_tracking(us::Vector{<:AbstractArray{T}}, ts::AbstractVector{T}, dx::T) where {T <: AbstractFloat}
    shock_locations = RDE.shock_locations.(us, dx)
    shock_counts = sum.(shock_locations)
    blocks_ixs = [1, (findall(diff(shock_counts) .!= 0) .+ 1)..., length(shock_counts) + 1]
    n_blocks = length(blocks_ixs) - 1

    shock_speeds = []
    for block in 1:n_blocks
        block_start_ix = blocks_ixs[block]
        block_end_ix = blocks_ixs[block + 1] - 1
        block_shock_locations = get_block_shock_data(shock_locations, blocks_ixs, block)
        n_shocks = sum(block_shock_locations[1])


        if n_shocks == 0 || length(block_shock_locations) == 1
            push!(shock_speeds, missing)
            continue
        end
        block_shock_indices = get_block_shock_indices(block_shock_locations)
        @assert length(block_shock_indices[1]) == length(block_shock_locations) "length(block_shock_indices[1]) ($(length(block_shock_indices[1]))) != length(block_shock_locations) ($(length(block_shock_locations)))"
        block_speeds = get_block_speeds(block_shock_indices, dx, ts[block_start_ix:block_end_ix])
        push!(shock_speeds, block_speeds)
    end
    #For blocks with no shocks or single point blocks, interpolate the speed from the neighboring blocks
    for block in 1:n_blocks
        if !ismissing(shock_speeds[block])
            continue
        end
        block_shock_locations = get_block_shock_data(shock_locations, blocks_ixs, block)
        block_n_ts = length(block_shock_locations)
        neighbor_speeds = []
        #search for previous block with non-missing speeds
        if block > 1
            prev_block = block - 1
            while prev_block > 0 && ismissing(shock_speeds[prev_block])
                prev_block -= 1
            end
            if prev_block > 0
                prev_block_speeds = shock_speeds[prev_block]
                prev_block_mean_speed = mean([last(shock_speed) for shock_speed in prev_block_speeds])
                push!(neighbor_speeds, prev_block_mean_speed)
            end
        end
        #search for next block with non-missing speeds
        if block < n_blocks
            next_block = block + 1
            while next_block <= n_blocks && ismissing(shock_speeds[next_block])
                next_block += 1
            end
            if next_block <= n_blocks
                next_block_speeds = shock_speeds[next_block]
                next_block_mean_speed = mean([first(shock_speed) for shock_speed in next_block_speeds])
                push!(neighbor_speeds, next_block_mean_speed)
            end
        end

        #if no neighboring blocks with non-missing speeds, set speed to zero
        if length(neighbor_speeds) == 0
            shock_speeds[block] = [zeros(T, block_n_ts)]
        else
            #interpolate the speed from the neighboring blocks
            interpolated_speed = mean(neighbor_speeds)
            shock_speeds[block] = [interpolated_speed * ones(T, block_n_ts)]
        end
    end
    return shock_speeds
end

function get_adjustment_speeds(block_speeds)
    first_shock_speeds = getindex.(block_speeds, 1)
    speeds = vcat(first_shock_speeds...)
    return speeds
end

function get_plotting_speed_adjustments(data, dx; max_speed = 4.6f0)
    block_speeds = speed_tracking(data, dx)
    plot_speeds = get_adjustment_speeds(block_speeds)[2:end]
    @debug "max_speed: $(maximum(plot_speeds)), min_speed: $(minimum(plot_speeds))"
    plot_speeds = adjust_for_jumps!(plot_speeds, max_speed)
    return plot_speeds
end

"""
    adjust_for_jumps!(plot_speeds, max_speed; fallback_speed = 1.71f0)
Sometimes speed blocks fail, e.g. if a single shock is dissappearing in the same frame as another is appearing.
Then it will appear as if there is a single shock that jumps to a new position.
Here we try to detect these jumps and adjust the entries with high speed. 
Adjustment is done by finding the nearby non-jump entries and averaging them.
If this fails, we set speed to `fallback_speed`.     
"""
function adjust_for_jumps!(plot_speeds, max_speed; fallback_speed = 1.71f0)
    all_indices = eachindex(plot_speeds)
    if any(s -> s > max_speed, plot_speeds)
        jump_indices = findall(s -> s > max_speed, plot_speeds)
        @debug "jump_indices: $jump_indices"
        @debug "jump_speeds: $(plot_speeds[jump_indices])"
        if !isempty(jump_indices)
            while !isempty(jump_indices)
                ix = popfirst!(jump_indices)
                non_jump_indices = setdiff(all_indices, jump_indices)
                previous_indices = non_jump_indices[non_jump_indices .< ix]
                next_indices = non_jump_indices[non_jump_indices .> ix]
                # @debug "previous_indices: $previous_indices"
                # @debug "next_indices: $next_indices"
                #find previous and next index with valid speed
                if isempty(previous_indices) || isempty(next_indices)
                    @warn "Cant correct for jump, setting speed at index $ix to $fallback_speed"
                    plot_speeds[ix] = fallback_speed
                else
                    prev_ix = last(previous_indices)
                    next_ix = first(next_indices)
                    @debug "ix: $ix, prev_ix: $prev_ix, next_ix: $next_ix"
                    prev_speed = plot_speeds[prev_ix]
                    next_speed = plot_speeds[next_ix]
                    adjusted_speed = (prev_speed + next_speed) / 2
                    @debug "jump_speed: $(plot_speeds[ix])"
                    @debug "prev_speed: $prev_speed, next_speed: $next_speed, adjusted_speed: $adjusted_speed"
                    plot_speeds[ix] = adjusted_speed
                end
            end
        end
    end
    return plot_speeds
end
